# Adapted from
# https://github.com/vllm-project/vllm/blob/aae6927be06dedbda39c6b0c30f6aa3242b84388/tests/entrypoints/openai/test_chat.py
import json
import os
import re
import tempfile

import jsonschema
import openai
import pytest
import yaml

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(scope="module")
def model_name():
    return "llama-3.1-model/Llama-3.1-8B-Instruct"


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file():
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "extra_llm_api_options.yaml")
    try:
        extra_llm_api_options_dict = {"guided_decoding_backend": "xgrammar"}

        with open(temp_file_path, 'w') as f:
            yaml.dump(extra_llm_api_options_dict, f)

        yield temp_file_path
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@pytest.fixture(scope="module")
def server(model_name: str, temp_extra_llm_api_options_file: str):
    model_path = get_model_path(model_name)

    # Use small max_batch_size/max_seq_len/max_num_tokens to avoid OOM on A10/A30 GPUs.
    args = [
        "--max_batch_size=8", "--max_seq_len=1024", "--max_num_tokens=1024",
        f"--extra_llm_api_options={temp_extra_llm_api_options_file}"
    ]
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


def test_json_schema(client: openai.OpenAI, model_name: str):
    json_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "pattern": "^[\\w]+$"
            },
            "population": {
                "type": "integer"
            },
        },
        "required": ["name", "population"],
    }
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role":
            "user",
            "content":
            "Give me the information of the capital of France in the JSON format.",
        },
    ]
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=256,
        response_format={
            "type": "json",
            "schema": json_schema
        },
    )

    message = chat_completion.choices[0].message
    assert message.content is not None
    assert message.role == "assistant"
    jsonschema.validate(json.loads(message.content), json_schema)


def test_json_schema_user_profile(client: openai.OpenAI, model_name: str):
    json_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The full name of the user."
            },
            "age": {
                "type": "integer",
                "description": "The age of the user, in years."
            },
        },
        "required": ["name", "age"],
    }
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role":
            "user",
            "content":
            f"Give an example JSON for an employee profile that fits this schema: {json_schema}",
        },
    ]
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=256,
        response_format={
            "type": "json",
            "schema": json_schema
        },
    )

    message = chat_completion.choices[0].message
    assert message.content is not None
    assert message.role == "assistant"
    first_json = json.loads(message.content)
    jsonschema.validate(first_json, json_schema)

    messages.extend([
        {
            "role": "assistant",
            "content": message.content,
        },
        {
            "role": "user",
            "content": "Give me another one with a different name and age.",
        },
    ])
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=256,
        response_format={
            "type": "json",
            "schema": json_schema
        },
    )

    message = chat_completion.choices[0].message
    assert message.content is not None
    assert message.role == "assistant"
    second_json = json.loads(message.content)
    jsonschema.validate(second_json, json_schema)

    assert (
        first_json["name"] != second_json["name"]
    ), "The model should have generated a different name in the second turn."
    assert (
        first_json["age"] != second_json["age"]
    ), "The model should have generated a different age in the second turn."


def test_regex(client: openai.OpenAI, model_name: str):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "What is the capital of France?",
        },
    ]
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=256,
        response_format={
            "type": "regex",
            "regex": "(Paris|London)"
        },
    )

    message = chat_completion.choices[0].message
    assert message.content is not None
    assert message.role == "assistant"
    assert re.match(r"(Paris|London)", message.content)


def test_ebnf(client: openai.OpenAI, model_name: str):
    ebnf_grammar = """
root ::= description
city ::= "London" | "Paris" | "Berlin" | "Rome"
description ::= city " is " status
status ::= "the capital of " country
country ::= "England" | "France" | "Germany" | "Italy"
"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful geography bot."
        },
        {
            "role": "user",
            "content": "Give me the information of the capital of France.",
        },
    ]
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=256,
        response_format={
            "type": "ebnf",
            "ebnf": ebnf_grammar
        },
    )

    message = chat_completion.choices[0].message
    assert message.content is not None
    assert message.role == "assistant"
    assert message.content == "Paris is the capital of France"


def test_structural_tag(client: openai.OpenAI, model_name: str):
    tool_get_current_weather = {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type":
                        "string",
                        "description":
                        "The city to find the weather for, e.g. 'San Francisco'",
                    },
                    "state": {
                        "type":
                        "string",
                        "description":
                        "the two-letter abbreviation for the state that the city is"
                        " in, e.g. 'CA' which would mean 'California'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    }

    tool_get_current_date = {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get the current date and time for a given timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type":
                        "string",
                        "description":
                        "The timezone to fetch the current date and time for, e.g. 'America/New_York'",
                    }
                },
                "required": ["timezone"],
            },
        },
    }

    system_prompt = f"""# Tool Instructions
- Always execute python code in messages that you share.
- When looking for real time information use relevant functions if available else fallback to brave_search
You have access to the following functions:
Use the function 'get_current_weather' to: Get the current weather in a given location
{tool_get_current_weather["function"]}
Use the function 'get_current_date' to: Get the current date and time for a given timezone
{tool_get_current_date["function"]}
If a you choose to call a function ONLY reply in the following format:
<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}
where
start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`
Here is an example,
<function=example_function_name>{{"example_name": "example_value"}}</function>
Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Always add your sources when using search results to answer the user query
You are a helpful assistant."""
    user_prompt = "You are in New York. Please get the current date and time, and the weather."

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=256,
        response_format={
            "type": "structural_tag",
            "format": {
                "type":
                "triggered_tags",
                "triggers": ["<function="],
                "tags": [
                    {
                        "begin": "<function=get_current_weather>",
                        "content": {
                            "type":
                            "json_schema",
                            "json_schema":
                            tool_get_current_weather["function"]["parameters"]
                        },
                        "end": "</function>",
                    },
                    {
                        "begin": "<function=get_current_date>",
                        "content": {
                            "type":
                            "json_schema",
                            "json_schema":
                            tool_get_current_date["function"]["parameters"]
                        },
                        "end": "</function>",
                    },
                ],
            },
        },
    )

    message = chat_completion.choices[0].message
    assert message.content is not None
    assert message.role == "assistant"

    match = re.search(r'<function=get_current_weather>([\S\s]+?)</function>',
                      message.content)
    params = json.loads(match.group(1))
    jsonschema.validate(params,
                        tool_get_current_weather["function"]["parameters"])

    match = re.search(r'<function=get_current_date>([\S\s]+?)</function>',
                      message.content)
    params = json.loads(match.group(1))
    jsonschema.validate(params, tool_get_current_date["function"]["parameters"])
