import os
import tempfile
from typing import Any

import openai
import pytest
import requests
import yaml
from utils.llm_data import llm_models_root

from ..test_llm import get_model_path
from .openai_server import RemoteMMEncoderServer

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(scope="module", ids=["Qwen2.5-VL-3B-Instruct"])
def model_name():
    return "Qwen2.5-VL-3B-Instruct"


@pytest.fixture(scope="module",
                params=[True, False],
                ids=["extra_options", "no_extra_options"])
def extra_encoder_options(request):
    return request.param


@pytest.fixture(scope="module")
def temp_extra_encoder_options_file(request):
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "extra_encoder_options.yaml")
    try:
        extra_encoder_options_dict = {
            "max_batch_size": 8,
            "max_num_tokens": 16384
        }

        with open(temp_file_path, 'w') as f:
            yaml.dump(extra_encoder_options_dict, f)

        yield temp_file_path
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@pytest.fixture(scope="module")
def server(model_name: str, extra_encoder_options: bool,
           temp_extra_encoder_options_file: str):
    model_path = get_model_path(model_name)
    args = ["--max_batch_size", "8"]
    if extra_encoder_options:
        args.extend(
            ["--extra_encoder_options", temp_extra_encoder_options_file])

    with RemoteMMEncoderServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteMMEncoderServer):
    return server.get_client()


@pytest.fixture(scope="module")
def async_client(server: RemoteMMEncoderServer):
    return server.get_async_client()


def test_multimodal_content_mm_encoder(
        client: openai.OpenAI,
        model_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:

    content_text = "Describe the natural environment in the image."
    image_url = str(llm_models_root() / "multimodals" / "test_data" /
                    "seashore.png")
    messages = [{
        "role":
        "user",
        "content": [{
            "type": "text",
            "text": content_text
        }, {
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }],
    }]

    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
    )
    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1
    choice = chat_completion.choices[0]
    # Verify mm_embedding_handle is present
    assert hasattr(choice, 'mm_embedding_handle')
    assert choice.mm_embedding_handle is not None
    # Verify the handle contains tensor information
    mm_handle = choice.mm_embedding_handle
    assert "tensor_size" in mm_handle
    assert mm_handle["tensor_size"][
        0] == 324  # qwen2.5-vl: 324 tokens for the same image
    assert mm_handle["tensor_size"][
        1] == 2048  # qwen2.5-vl: hidden_size of the vision encoder

    return messages, mm_handle  # used by tests/unittest/llmapi/apps/_test_openai_chat_multimodal.py


def test_health(server: RemoteMMEncoderServer):
    health_url = server.url_for("health")
    response = requests.get(health_url)
    assert response.status_code == 200


def test_models_endpoint(client: openai.OpenAI, model_name: str):
    models = client.models.list()
    assert len(models.data) >= 1

    model_names = [model.id for model in models.data]
    # The model name might be transformed, so check if any model contains our base name
    expected_name = model_name.split('/')[-1]
    assert any(expected_name in name for name in model_names)
