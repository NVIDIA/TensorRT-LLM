import json
import os
import subprocess
import sys
import tempfile

import pytest
import yaml

from .openai_server import RemoteOpenAIServer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_llm import get_model_path


@pytest.fixture(scope="module", ids=["TinyLlama-1.1B-Chat"])
def model_name():
    return "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


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
    # fix port to facilitate concise trtllm-serve examples
    args = ["--extra_llm_api_options", temp_extra_llm_api_options_file]
    with RemoteOpenAIServer(model_path, args, port=8000) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def example_root():
    llm_root = os.getenv("LLM_ROOT")
    return os.path.join(llm_root, "examples", "serve")


@pytest.mark.parametrize(
    "exe, script", [("python3", "openai_chat_client.py"),
                    ("python3", "openai_completion_client.py"),
                    ("python3", "openai_completion_client_json_schema.py"),
                    ("bash", "curl_chat_client.sh"),
                    ("bash", "curl_completion_client.sh"),
                    ("bash", "genai_perf_client.sh")])
def test_trtllm_serve_examples(exe: str, script: str,
                               server: RemoteOpenAIServer, example_root: str):
    client_script = os.path.join(example_root, script)
    # CalledProcessError will be raised if any errors occur
    result = subprocess.run([exe, client_script],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=True)
    if script.startswith("curl"):
        # For curl scripts, we expect a JSON response
        result_stdout = result.stdout.strip()
        try:
            data = json.loads(result_stdout)
            assert "code" not in data or data[
                "code"] == 200, f"Unexpected response: {data}"
        except json.JSONDecodeError as e:
            pytest.fail(
                f"Failed to parse JSON response from {script}: {e}\nStdout: {result_stdout}\nStderr: {result.stderr}"
            )
