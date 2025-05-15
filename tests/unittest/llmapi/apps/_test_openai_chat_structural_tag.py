# Adapted from
# https://github.com/vllm-project/vllm/blob/aae6927be06dedbda39c6b0c30f6aa3242b84388/tests/entrypoints/openai/test_chat.py
import os
import tempfile

import openai
import pytest
import yaml

from ..test_llm import get_model_path, similar
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(scope="module", ids=["TinyLlama-1.1B-Chat"])
def model_name():
    return "llama-3.1-model/Llama-3.1-8B-Instruct"


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file(request):
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "extra_llm_api_options.yaml")
    try:
        extra_llm_api_options_dict = {
            "guided_decoding_backend": "xgrammar",
            "pytorch_backend_config": {
                "disable_overlap_scheduler": True,
            }
        }

        with open(temp_file_path, 'w') as f:
            yaml.dump(extra_llm_api_options_dict, f)

        yield temp_file_path
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@pytest.fixture(scope="module")
def server(model_name: str, temp_extra_llm_api_options_file: str):
    model_path = get_model_path(model_name)
    args = [
        "--backend", "pytorch", "--extra_llm_api_options",
        temp_extra_llm_api_options_file
    ]
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


@pytest.fixture(scope="module")
def tool_get_current_weather():
    return {
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


@pytest.fixture(scope="module")
def tool_get_current_date():
    return {
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


def test_chat_structural_tag(client: openai.OpenAI, model_name: str,
                             tool_get_current_weather, tool_get_current_date):
    messages = [
        {
            "role":
            "system",
            "content":
            f"""
# Tool Instructions
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
You are a helpful assistant.""",
        },
        {
            "role":
            "user",
            "content":
            "You are in New York. Please get the current date and time, and the weather.",
        },
    ]

    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=100,
        response_format={
            "type":
            "structural_tag",
            "structures": [
                {
                    "begin": "<function=get_current_weather>",
                    "schema":
                    tool_get_current_weather["function"]["parameters"],
                    "end": "</function>",
                },
                {
                    "begin": "<function=get_current_date>",
                    "schema": tool_get_current_date["function"]["parameters"],
                    "end": "</function>",
                },
            ],
            "triggers": ["<function="],
        },
    )
    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1
    message = chat_completion.choices[0].message
    assert message.content is not None
    assert message.role == "assistant"

    reference = '<function=get_current_date>{"timezone": "America/New_York"}</function>\n<function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function>\n\nSources:\n- get_current_date function\n- get_current_weather function'
    assert similar(chat_completion.choices[0].message.content, reference)
