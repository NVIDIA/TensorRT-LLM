import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml
from _model_test_utils import _hf_model_dir_or_hub_id
from click.testing import CliRunner
from utils.cpp_paths import llm_root  # noqa: F401

from tensorrt_llm.commands.bench import main


def tiny_llama_details():
    model_path = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_path_or_name = _hf_model_dir_or_hub_id(model_path, model_name)
    return model_path_or_name, model_name, model_path


def run_benchmark(model_name: str, dataset_path: str, extra_llm_api_options_path: str):
    runner = CliRunner()

    args = [
        "--model",
        model_name,
        "throughput",
        "--backend",
        "_autodeploy",
        "--dataset",
        dataset_path,
        "--extra_llm_api_options",
        f"{extra_llm_api_options_path}",
    ]
    result = runner.invoke(main, args, catch_exceptions=False)
    assert result.exit_code == 0


def prepare_dataset(root_dir: str, temp_dir: str, model_path_or_name: str):
    _DATASET_NAME = "synthetic_128_128.txt"
    dataset_path = Path(temp_dir, _DATASET_NAME)
    dataset_tool = Path(root_dir, "benchmarks", "cpp", "prepare_dataset.py")
    script_dir = Path(root_dir, "benchmarks", "cpp")

    # Generate a small dataset to run a test - matching workload configuration
    command = [
        "python3",
        f"{dataset_tool}",
        "--stdout",
        "--tokenizer",
        model_path_or_name,
        "token-norm-dist",
        "--input-mean",
        "128",
        "--output-mean",
        "128",
        "--input-stdev",
        "0",
        "--output-stdev",
        "0",
        "--num-requests",
        "10",
    ]
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(
        command, cwd=str(script_dir), capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to prepare dataset: {result.stderr}")
    # Grab the stdout and write it to a dataset file for passing to suite.
    with open(dataset_path, "w") as dataset:
        dataset.write(result.stdout)
    return dataset_path


@pytest.mark.parametrize("compile_backend", ["torch-compile", "torch-opt", "torch-cudagraph"])
def test_trtllm_bench(llm_root, compile_backend):  # noqa: F811
    model_path_or_name, model_name, model_path = tiny_llama_details()
    with tempfile.TemporaryDirectory() as temp_dir:
        extra_llm_api_options_path = f"{temp_dir}/extra_llm_api_options.yaml"
        with open(extra_llm_api_options_path, "w") as f:
            yaml.dump(
                {
                    "model_kwargs": {"num_hidden_layers": 2},
                    "cuda_graph_batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
                    "max_batch_size": 128,
                    "compile_backend": compile_backend,
                },
                f,
            )

        dataset_path = prepare_dataset(llm_root, temp_dir, model_path_or_name)
        run_benchmark(model_name, dataset_path, extra_llm_api_options_path)
