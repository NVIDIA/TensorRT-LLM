import os
import subprocess
import sys

import pytest

from .openai_server import RemoteOpenAIServer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_llm import get_model_path


@pytest.fixture(scope="module", params=["LLM", "Multimodal"])
def model_name(request):
    model_map = {
        "LLM": "TinyLlama-1.1B-Chat",
        "Multimodal": "Qwen2-VL-7B-Instruct"
    }
    return model_map[request.param]


@pytest.fixture(scope="module")
def server(model_name: str):
    model_path_map = {
        "TinyLlama-1.1B-Chat": "llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
        "Qwen2-VL-7B-Instruct": "Qwen2-VL-7B-Instruct"
    }
    model_path = get_model_path(model_path_map[model_name])
    # fix port to facilitate concise trtllm-serve examples
    server_kwargs = {"port": 8000}

    # Add multimodal and model-specific configurations
    if "Qwen2-VL-7B-Instruct" in model_name:
        import yaml
        config = {'kv_cache_config': {'enable_block_reuse': False}}
        with open('extra-llm-api-config.yml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        server_kwargs.update({
            "cli_args": [
                "--backend", "pytorch", "--extra_llm_api_options",
                "extra-llm-api-config.yml"
            ]
        })

    with RemoteOpenAIServer(model_path, **server_kwargs) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def example_root():
    llm_root = os.getenv("LLM_ROOT")
    return os.path.join(llm_root, "examples", "serve")


def test_trtllm_serve_examples(model_name: str, example_root: str,
                               server: RemoteOpenAIServer):
    # Define scripts based on model
    if "Qwen2-VL-7B-Instruct" in model_name:
        scripts = [
            ("python3", "openai_chat_client_for_multimodal.py"),
            ("bash", "curl_chat_client_for_multimodal.sh"),
        ]
    else:  # TinyLlama
        scripts = [
            ("python3", "openai_chat_client.py"),
            ("python3", "openai_completion_client.py"),
            ("bash", "curl_chat_client.sh"),
            ("bash", "curl_completion_client.sh"),
        ]

    for exe, script in scripts:
        client_script = os.path.join(example_root, script)
        # CalledProcessError will be raised if any errors occur
        subprocess.run([exe, client_script],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       text=True,
                       check=True)
