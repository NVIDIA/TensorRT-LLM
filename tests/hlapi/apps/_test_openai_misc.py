import os
import sys

import openai
import pytest
import requests
from openai_server import RemoteOpenAIServer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_llm import get_model_path

from tensorrt_llm.version import __version__ as VERSION


@pytest.fixture(scope="module")
def model_name():
    return "llama-models-v3/llama-v3-8b-instruct-hf"


@pytest.fixture(scope="module")
def server(model_name: str):
    model_path = get_model_path(model_name)
    args = ["--max_beam_width", "4"]
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


def test_version(server: RemoteOpenAIServer):
    version_url = server.url_for("version")
    response = requests.get(version_url)
    assert response.status_code == 200
    assert response.json()["version"] == VERSION


def test_health(server: RemoteOpenAIServer):
    health_url = server.url_for("health")
    response = requests.get(health_url)
    assert response.status_code == 200


def test_model(client: openai.OpenAI, model_name: str):
    model = client.models.list().data[0]
    assert model.id == model_name.split('/')[-1]
