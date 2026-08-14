import os
import subprocess
import sys
import tempfile

import pytest
import yaml
from utils.util import skip_gpu_memory_less_than_80gb

from .openai_server import RemoteOpenAIServer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_llm import get_model_path


@pytest.fixture(scope="module", ids=["Qwen2.5-VL-3B-Instruct"])
def model_name():
    return "Qwen2.5-VL-3B-Instruct"


@pytest.fixture(scope="module")
def model_path(model_name: str):
    return get_model_path(model_name)


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file(request):
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(
        temp_dir, "extra_llm_api_options_multimodal_benchmark.yaml")
    try:
        extra_llm_api_options_dict = {
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.6,
            },
            "max_num_tokens": 16384,  # for pytorch backend
            # NOTE: This is for video support.
            "build_config": {
                "max_num_tokens": 16384,
            }
        }

        with open(temp_file_path, 'w') as f:
            yaml.dump(extra_llm_api_options_dict, f)

        yield temp_file_path
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@pytest.fixture(scope="module")
def server(model_path: str, temp_extra_llm_api_options_file: str):
    # Use pytorch backend for multimodal support and fix port to facilitate benchmarking
    args = ["--extra_llm_api_options", temp_extra_llm_api_options_file]
    with RemoteOpenAIServer(model_path, port=8000,
                            cli_args=args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def benchmark_root():
    llm_root = os.getenv("LLM_ROOT")
    return os.path.join(llm_root, "tensorrt_llm", "serve", "scripts")


# TODO: Add this to /code/llm-models
def vision_arena_dataset_path():
    """Return a vision arena dataset path for testing."""
    return "lmarena-ai/vision-arena-bench-v0.1"


@skip_gpu_memory_less_than_80gb
@pytest.mark.parametrize("dataset_name,dataset_args",
                         [("random_image", {
                             "--random-num-images": "1",
                             "--random-image-size": "512",
                         }),
                          ("random_image", {
                              "--random-num-images": "2",
                              "--random-image-size": "512",
                          }),
                          ("hf", {
                              "--dataset-path": vision_arena_dataset_path(),
                          })],
                         ids=[
                             "random_image-single_image",
                             "random_image-dual_images",
                             "hf-vision_arena_dataset"
                         ])
def test_trtllm_serve_multimodal_benchmark(server: RemoteOpenAIServer,
                                           benchmark_root: str, model_path: str,
                                           dataset_name: str,
                                           dataset_args: dict):
    """Test multimodal benchmark serving with different datasets."""
    client_script = os.path.join(benchmark_root, "benchmark_serving.py")

    # Base command arguments
    benchmark_cmd = [
        "python3",
        client_script,
        "--backend",
        "openai-chat",  # Required for multimodal
        "--dataset-name",
        dataset_name,
        "--model",
        "qwen2.5-vl",
        "--tokenizer",
        model_path,
        "--num-prompts",
        "10",  # Small number for testing
    ]

    # Add dataset-specific arguments
    for key, value in dataset_args.items():
        benchmark_cmd.extend([key, str(value)])

    # CalledProcessError will be raised if any errors occur
    result = subprocess.run(benchmark_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=True)

    # Basic validation that the benchmark ran successfully
    assert result.returncode == 0
    assert "Serving Benchmark Result" in result.stdout
