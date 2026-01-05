import json

import openai
import pytest
from openai.types.responses import (ResponseCompletedEvent,
                                    ResponseReasoningTextDeltaEvent,
                                    ResponseTextDeltaEvent)

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(scope="module",
                params=[
                    "gpt_oss/gpt-oss-20b", "DeepSeek-R1-Distill-Qwen-1.5B",
                    "Qwen3/Qwen3-0.6B"
                ])
def model(request):
    return request.param


@pytest.fixture(scope="module",
                params=[0, 2],
                ids=["disable_processpool", "enable_processpool"])
def num_postprocess_workers(request):
    return request.param


@pytest.fixture(scope="module")
def server(model: str, num_postprocess_workers: int):
    model_path = get_model_path(model)

    args = ["--num_postprocess_workers", f"{num_postprocess_workers}"]
    if model.startswith("Qwen3"):
        args.extend(["--reasoning_parser", "qwen3"])
    elif model.startswith("DeepSeek-R1"):
        args.extend(["--reasoning_parser", "deepseek-r1"])

    if not model.startswith("gpt_oss"):
        args.extend(["--tool_parser", "qwen3"])

    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_async_client()


def check_reponse(response, prefix=""):
    print(f"response: {response}")
    reasoning_exist, message_exist = False, False
    for output in response.output:
        if output.type == "reasoning":
            reasoning_exist = True
        elif output.type == "message":
            message_exist = True

    assert reasoning_exist, f"{prefix}Reasoning content not exists!"
    assert message_exist, f"{prefix}Message content not exists!"


def check_tool_calling(response, first_resp=True, prefix=""):
    print(f"response: {response}")
    reasoning_exist, tool_call_exist, message_exist = False, False, False
    reasoning_content, message_content = "", ""
    function_call = None
    for output in response.output:
        if output.type == "reasoning":
            reasoning_exist = True
            reasoning_content = output.content[0].text
        elif output.type == "function_call":
            tool_call_exist = True
            function_call = output
        elif output.type == "message":
            message_exist = True
            message_content = output.content[0].text

    err_msg = f"{prefix}Invalid tool calling {'1st' if first_resp else '2nd'} response:"
    if first_resp:
        assert reasoning_exist, f"{err_msg} reasoning content not exists! ({reasoning_content})"
        assert tool_call_exist, f"{err_msg} tool call content not exists! ({function_call})"
        assert not message_exist, f"{err_msg} message content should not exist! ({message_content})"

        return function_call
    else:
        assert reasoning_exist, f"{err_msg} reasoning content not exists! ({reasoning_content})"
        assert message_exist, f"{err_msg} message content not exists! ({message_content})"
        assert not tool_call_exist, f"{err_msg} tool call content should not exist! ({function_call})"


def _get_qwen3_nothink_input(model: str, input: str):
    return f"{input} /no_think" if model.startswith("Qwen3") else input


@pytest.mark.asyncio(loop_scope="module")
async def test_reasoning(client: openai.AsyncOpenAI, model: str):
    response = await client.responses.create(
        model=model,
        input="Which one is larger as numeric, 9.9 or 9.11?",
    )

    check_reponse(response, "test_reasoning: ")


@pytest.mark.asyncio(loop_scope="module")
async def test_reasoning_effort(client: openai.AsyncOpenAI, model: str):
    for effort in ["low", "medium", "high"]:
        response = await client.responses.create(
            model=model,
            instructions="Use less than 1024 tokens for the whole response",
            input="Which one is larger as numeric, 9.9 or 9.11?",
            reasoning={"effort": effort},
        )
        check_reponse(response, f"test_reasoning_effort_{effort}: ")


@pytest.mark.asyncio(loop_scope="module")
async def test_chat(client: openai.AsyncOpenAI, model: str):
    response = await client.responses.create(
        model=model,
        input=[{
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
        }],
    )
    check_reponse(response, "test_chat: ")


@pytest.mark.asyncio(loop_scope="module")
async def test_multi_turn_chat(client: openai.AsyncOpenAI, model: str,
                               num_postprocess_workers: int):
    if num_postprocess_workers > 0:
        pytest.skip(
            "Response store is disabled when num_postprocess_workers > 0")

    response = await client.responses.create(
        model=model,
        input=_get_qwen3_nothink_input(model, "What is the answer of 1+1?"),
    )
    check_reponse(response, "test_multi_turn_chat_1: ")

    response_2 = await client.responses.create(
        model=model,
        input=_get_qwen3_nothink_input(
            model, "What is the answer of previous question?"),
        previous_response_id=response.id,
    )
    check_reponse(response_2, "test_multi_turn_chat_2: ")


def get_current_weather(location: str, format: str = "celsius") -> dict:
    return {"sunny": True, "temperature": 20 if format == "celsius" else 68}


@pytest.mark.asyncio(loop_scope="module")
async def test_tool_calls(client: openai.AsyncOpenAI, model: str):
    if model.startswith("DeepSeek-R1"):
        pytest.skip("DeepSeek-R1 does not support tool calls")

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

    response = await client.responses.create(
        model=model,
        input=messages,
        tools=[tool_get_current_weather],
    )

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
    if model.startswith("DeepSeek-R1"):
        pytest.skip("DeepSeek-R1 does not support tool calls")

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

    assert function_call is not None, "function call not exists!"

    reasoning = "".join(reasoning_deltas)
    tool_args = json.loads(function_call.arguments)

    assert function_call.name == "get_current_weather", "wrong function calling name"
    assert tool_args, "tool args not exists!"
    assert reasoning, "reasoning not exists!"

    get_current_weather(**tool_args)
