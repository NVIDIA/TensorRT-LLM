import os
import subprocess
import sys
import tempfile

import pytest
import yaml

from .openai_server import RemoteOpenAIServer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_llm import get_model_path


@pytest.fixture(scope="module", ids=["Qwen2.5-VL-3B-Instruct"])
def model_name():
    return "Qwen2.5-VL-3B-Instruct"


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file(request):
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "extra_llm_api_options.yaml")
    try:
        extra_llm_api_options_dict = {
            "kv_cache_config": {
                "enable_block_reuse": False,
                "free_gpu_memory_fraction": 0.6,
            },
            "max_num_tokens": 16384,  # for pytorch backend
            # NOTE: This is for video support.
            "build_config": {
                "max_num_tokens": 16384,
            },
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
    args = ["--extra_llm_api_options", temp_extra_llm_api_options_file]
    with RemoteOpenAIServer(model_path, port=8000,
                            cli_args=args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def example_root():
    llm_root = os.getenv("LLM_ROOT")
    return os.path.join(llm_root, "examples", "serve")


@pytest.mark.parametrize("exe, script",
                         [("python3", "openai_chat_client_for_multimodal.py"),
                          ("bash", "genai_perf_client_for_multimodal.sh")])
def test_trtllm_serve_examples(exe: str, script: str,
                               server: RemoteOpenAIServer, example_root: str):
    client_script = os.path.join(example_root, script)
    # CalledProcessError will be raised if any errors occur
    subprocess.run([exe, client_script],
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE,
                   text=True,
                   check=True)
