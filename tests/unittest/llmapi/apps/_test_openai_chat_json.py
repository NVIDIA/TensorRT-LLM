# Adapted from
# https://github.com/vllm-project/vllm/blob/aae6927be06dedbda39c6b0c30f6aa3242b84388/tests/entrypoints/openai/test_chat.py
import json
import os
import tempfile
from typing import Any

import jsonschema
import openai
import pytest
import yaml

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(scope="module", ids=["TinyLlama-1.1B-Chat"])
def model_name():
    return "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file(request):
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "extra_llm_api_options.yaml")
    try:
        extra_llm_api_options_dict = {
            "guided_decoding_backend": "xgrammar",
            "disable_overlap_scheduler":
            True,  # Guided decoding is not supported with overlap scheduler
        }

        with open(temp_file_path, "w") as f:
            yaml.dump(extra_llm_api_options_dict, f)

        yield temp_file_path
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@pytest.fixture(scope="module")
def server(model_name: str, temp_extra_llm_api_options_file: str):
    model_path = get_model_path(model_name)
    args = ["--extra_llm_api_options", temp_extra_llm_api_options_file]
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


@pytest.fixture(scope="module")
def user_profile_schema():
    """Provides a sample JSON schema for a user profile."""
    return {
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


def test_chat_json_schema(client: openai.OpenAI, model_name: str,
                          user_profile_schema):
    """
    Tests the `json` response format in a multi-turn synchronous conversation.
    Adapted from https://github.com/vllm-project/vllm/blob/aae6927be06dedbda39c6b0c30f6aa3242b84388/tests/entrypoints/openai/test_chat.py#L413
    """

    def _create_and_validate_response(
            messages: list[dict[str, Any]]) -> dict[str, Any]:
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1000,
            temperature=0.0,
            response_format={
                "type": "json",
                "schema": user_profile_schema
            },
        )
        message = chat_completion.choices[0].message
        assert message.content is not None
        try:
            message_json = json.loads(message.content)
        except json.JSONDecodeError:
            pytest.fail(
                f"The output was not a valid JSON string. Output: {message.content}"
            )

        jsonschema.validate(instance=message_json, schema=user_profile_schema)
        return message_json, message.content

    messages = [
        {
            "role": "system",
            "content": "you are a helpful assistant"
        },
        {
            "role":
            "user",
            "content":
            f"Give an example JSON for an employee profile that fits this schema: {user_profile_schema}",
        },
    ]
    first_json, first_content = _create_and_validate_response(messages)
    messages.extend([
        {
            "role": "assistant",
            "content": first_content,
        },
        {
            "role": "user",
            "content": "Give me another one with a different name and age.",
        },
    ])
    second_json, second_content = _create_and_validate_response(messages)

    assert (
        first_json["name"] != second_json["name"]
    ), "The model should have generated a different name in the second turn."
    assert (
        first_json["age"] != second_json["age"]
    ), "The model should have generated a different age in the second turn."
