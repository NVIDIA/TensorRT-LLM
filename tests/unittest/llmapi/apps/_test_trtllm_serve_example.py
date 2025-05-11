import os
import subprocess
import sys

import pytest

from .openai_server import RemoteOpenAIServer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_llm import get_model_path


@pytest.fixture(scope="module", ids=["TinyLlama-1.1B-Chat"])
def model_name():
    return "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module")
def server(model_name: str):
    model_path = get_model_path(model_name)
    # fix port to facilitate concise trtllm-serve examples
    with RemoteOpenAIServer(model_path, port=8000) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def example_root():
    llm_root = os.getenv("LLM_ROOT")
    return os.path.join(llm_root, "examples", "serve")


@pytest.mark.parametrize("exe, script",
                         [("python3", "openai_chat_client.py"),
                          ("python3", "openai_completion_client.py"),
                          ("bash", "curl_chat_client.sh"),
                          ("bash", "curl_completion_client.sh"),
                          ("bash", "genai_perf_client.sh")])
def test_trtllm_serve_examples(exe: str, script: str,
                               server: RemoteOpenAIServer, example_root: str):
    client_script = os.path.join(example_root, script)
    # CalledProcessError will be raised if any errors occur
    subprocess.run([exe, client_script],
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE,
                   text=True,
                   check=True)
