import os
import tempfile

import openai
import pytest
import yaml
from pydantic import BaseModel, Field

from ..test_llm import get_model_path
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
def capital_info_model():

    class CapitalInfo(BaseModel):
        name: str = Field(...,
                          pattern=r"^\w+$",
                          description="The name of the capital city")
        population: int = Field(...,
                                description="The population of the capital city")

    return CapitalInfo


def test_chat_json_schema(client: openai.OpenAI, model_name: str,
                          capital_info_model):

    CapitalInfo = capital_info_model
    messages = [{
        "role":
        "user",
        "content":
        "Please generate the information of the capital of France in the JSON format. ",
    }, ]

    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": CapitalInfo.model_json_schema(),
        },
        temperature=0.7,
        max_completion_tokens=100,
    )

    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1
    message = chat_completion.choices[0].message
    assert message.content is not None
    assert message.role == "assistant"

    capital_info = CapitalInfo.model_validate_json(message.content)

    assert isinstance(capital_info, CapitalInfo)
    assert capital_info.name == "Paris"
    assert isinstance(capital_info.population, int)
    assert capital_info.population > 0
