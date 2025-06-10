import subprocess
import tempfile

import yaml
from _model_test_utils import hf_model_dir_or_hub_id
from click.testing import CliRunner
from utils.llm_data import llm_models_root

from tensorrt_llm.commands.bench import main as trtllm_bench


def prepare_dataset(temp_dir, model_name):
    _DATASET_NAME = "synthetic_128_128.txt"
    dataset_path = f"{temp_dir}/{_DATASET_NAME}"
    with open(f"{temp_dir}/{_DATASET_NAME}", "w") as outfile:
        subprocess.run(
            [
                "python",
                "benchmarks/cpp/prepare_dataset.py",
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
            ],
            stdout=outfile,
            check=True,
        )
    return dataset_path


def test_trtllm_bench():
    model_name = hf_model_dir_or_hub_id(
        f"{llm_models_root()}/Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        with open(f"{temp_dir}/model_kwargs.yaml", "w") as f:
            yaml.dump(
                {"model_kwargs": {"num_hidden_layers": 2}, "cuda_graph_batch_sizes": [1, 2]}, f
            )

        dataset_path = prepare_dataset(temp_dir, model_name)
        runner = CliRunner()
        result = runner.invoke(
            trtllm_bench,
            [
                "--model",
                model_name,
                "throughput",
                "--backend",
                "_autodeploy",
                "--dataset",
                dataset_path,
                "--extra_llm_api_options",
                f"{temp_dir}/model_kwargs.yaml",
            ],
        )
        assert result.exit_code == 0
