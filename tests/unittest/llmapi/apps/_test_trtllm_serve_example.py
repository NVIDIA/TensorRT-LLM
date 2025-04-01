import os
import subprocess
import sys

import pytest

from .openai_server import RemoteOpenAIServer

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "examples",
                 "serve"))
from openai_chat_client import run_chat
from openai_completion_client import run_completion

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_llm import get_model_path


@pytest.fixture(scope="module", ids=["TinyLlama-1.1B-Chat"])
def model_name():
    return "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module")
def server(model_name: str):
    model_path = get_model_path(model_name)
    with RemoteOpenAIServer(model_path) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def base_url(server: RemoteOpenAIServer):
    return server.url_for("v1")


@pytest.fixture(scope="module")
def example_root():
    llm_root = os.getenv("LLM_ROOT")
    return os.path.join(llm_root, "examples", "serve")


def test_openai_chat_client(base_url: str):
    response = run_chat(max_completion_tokens=10, base_url=base_url)
    assert response.choices[0].message.content is not None


def test_openai_completion_client(base_url: str):
    response = run_completion(max_tokens=10, base_url=base_url)
    assert response.choices[0].text is not None


def test_curl_chat_client(example_root: str, base_url: str):
    client_script = os.path.join(example_root, "curl_chat_client.sh")
    # CalledProcessError will be raised if any errors occur
    subprocess.run(["bash", client_script],
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE,
                   env={"BASE_URL": base_url},
                   text=True,
                   check=True)


def test_curl_completion_client(example_root: str, base_url: str):
    client_script = os.path.join(example_root, "curl_completion_client.sh")
    # CalledProcessError will be raised if any errors occur
    subprocess.run(["bash", client_script],
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE,
                   env={"BASE_URL": base_url},
                   text=True,
                   check=True)
