import os
import subprocess
import sys

import pytest
from transformers import AutoTokenizer
from utils.util import skip_gpu_memory_less_than_80gb

from tensorrt_llm.serve.scripts.benchmark_dataset import RandomDataset

from .openai_server import RemoteOpenAIServer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_llm import get_model_path


@pytest.fixture(scope="module")
def model_name():
    return "llama-3.1-model/Meta-Llama-3.1-8B"


@pytest.fixture(scope="module")
def model_path(model_name: str):
    return get_model_path(model_name)


@pytest.fixture(scope="module")
def server(model_path: str):
    # fix port to facilitate concise trtllm-serve examples
    with RemoteOpenAIServer(model_path, port=8000) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def benchmark_root():
    llm_root = os.getenv("LLM_ROOT")
    return os.path.join(llm_root, "tensorrt_llm", "serve", "scripts")


def dataset_path(dataset_name: str):
    if dataset_name == "sharegpt":
        return get_model_path(
            "datasets/ShareGPT_V3_unfiltered_cleaned_split.json")
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")


@skip_gpu_memory_less_than_80gb
def test_trtllm_serve_benchmark(server: RemoteOpenAIServer, benchmark_root: str,
                                model_path: str):
    client_script = os.path.join(benchmark_root, "benchmark_serving.py")
    dataset = dataset_path("sharegpt")
    benchmark_cmd = [
        "python3", client_script, "--dataset-name", "sharegpt", "--model",
        "llama", "--dataset-path", dataset, "--tokenizer", model_path
    ]

    # CalledProcessError will be raised if any errors occur
    subprocess.run(benchmark_cmd,
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE,
                   text=True,
                   check=True)


@pytest.mark.parametrize("prefix_length", [0, 10, 100])
@pytest.mark.parametrize("model_name", [
    "llama-3.1-model/Meta-Llama-3.1-8B", "DeepSeek-R1/DeepSeek-R1",
    "Qwen2-7B-Instruct", "gpt_oss/gpt-oss-20b"
])
def test_trtllm_serve_benchmark_with_chat_template(prefix_length: int,
                                                   model_name: str):
    dataset = RandomDataset(
        sample_from_sharegpt=False,
        return_text=True,
        random_seed=0,
    )

    tokenizer = AutoTokenizer.from_pretrained(get_model_path(model_name))
    dataset.sample(tokenizer,
                   10,
                   use_chat_template=True,
                   prefix_len=prefix_length)

    # Check that all returned SampleRequest instances have the correct input length
    sample_requests = dataset.sample(tokenizer,
                                     10,
                                     prefix_len=prefix_length,
                                     use_chat_template=True)
    for req in sample_requests:
        # The prompt_len attribute should match the expected input length
        prompt_len = RandomDataset.DEFAULT_INPUT_LEN + prefix_length
        assert req.prompt_len <= prompt_len, (
            f"SampleRequest prompt_len ({req.prompt_len} + {prefix_length}) exceeds the default input length {RandomDataset.DEFAULT_INPUT_LEN}"
        )
        input_ids = tokenizer.encode(req.prompt)["input_ids"]
        assert len(input_ids) <= prompt_len, (
            f"SampleRequest prompt length ({len(input_ids)}) exceeds the default input length {RandomDataset.DEFAULT_INPUT_LEN}"
        )
