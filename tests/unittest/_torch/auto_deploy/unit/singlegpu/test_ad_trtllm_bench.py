import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml
from _model_test_utils import _hf_model_dir_or_hub_id
from click.testing import CliRunner
from utils.cpp_paths import llm_root  # noqa: F401
from utils.llm_data import llm_models_root

from tensorrt_llm.commands.bench import main


def prepare_dataset(root_dir: str, temp_dir: str, model_name: str):
    _DATASET_NAME = "synthetic_128_128.txt"
    dataset_path = Path(temp_dir, _DATASET_NAME)
    dataset_tool = Path(root_dir, "benchmarks", "cpp", "prepare_dataset.py")
    script_dir = Path(root_dir, "benchmarks", "cpp")

    # Generate a small dataset to run a test.
    command = [
        "python3",
        f"{dataset_tool}",
        "--stdout",
        "--tokenizer",
        model_name,
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
    result = subprocess.run(command, cwd=str(script_dir), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to prepare dataset: {result.stderr}")
    # Grab the stdout and write it to a dataset file for passing to suite.
    with open(dataset_path, "w") as dataset:
        dataset.write(result.stdout)
    return dataset_path


def run_benchmark(model_name: str, dataset_path: str, temp_dir: str):
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
        f"{temp_dir}/model_kwargs.yaml",
    ]
    result = runner.invoke(main, args, catch_exceptions=False)
    assert result.exit_code == 0


@pytest.mark.skip("https://nvbugswb.nvidia.com/NVBugs5/redir.aspx?url=/5443039")
def test_trtllm_bench(llm_root):  # noqa: F811
    model_name = _hf_model_dir_or_hub_id(
        f"{llm_models_root()}/TinyLlama-1.1B-Chat-v1.0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        with open(f"{temp_dir}/model_kwargs.yaml", "w") as f:
            yaml.dump(
                {
                    "model_kwargs": {"num_hidden_layers": 2},
                    "cuda_graph_batch_sizes": [1, 2],
                    "max_batch_size": 128,
                },
                f,
            )

        dataset_path = prepare_dataset(llm_root, temp_dir, model_name)
        run_benchmark(model_name, dataset_path, temp_dir)
