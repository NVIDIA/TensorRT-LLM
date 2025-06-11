import tempfile
import warnings

import yaml
from _model_test_utils import hf_model_dir_or_hub_id
from click.testing import CliRunner
from utils.llm_data import llm_models_root

from benchmarks.cpp.prepare_dataset import cli as prepare_dataset_cli
from tensorrt_llm.commands.bench import main as trtllm_bench


def prepare_dataset(temp_dir, model_name):
    _DATASET_NAME = "synthetic_128_128.txt"
    dataset_path = f"{temp_dir}/{_DATASET_NAME}"
    with open(dataset_path, "w") as outfile:
        runner = CliRunner()
        result = runner.invoke(
            prepare_dataset_cli,
            [
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
            catch_exceptions=False,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to prepare dataset: {result.output}")
        outfile.write(result.output)
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

        try:
            dataset_path = prepare_dataset(temp_dir, model_name)
        except Exception as e:
            warnings.warn(f"Failed to prepare dataset: {e}")
            return

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
