"""Testing build_and_run_ad end2end."""

from typing import Dict

import pytest
from _model_test_utils import get_small_model_config_pytest_param
from build_and_run_ad import ExperimentConfig, main


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("mode", ["graph", "transformers"])
@pytest.mark.parametrize(
    "experiment_config",
    [
        get_small_model_config_pytest_param(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            attn_backend="flashinfer",
            compile_backend="torch-opt",
        ),
    ],
)
def test_build_ad(world_size: int, experiment_config: Dict, mode: str):
    experiment_config["args"]["world_size"] = world_size
    experiment_config["args"]["runtime"] = "trtllm"  # Default runtime set to trtllm
    experiment_config["args"]["mode"] = mode
    experiment_config = ExperimentConfig(**experiment_config)
    print(f"Experiment Config: {experiment_config}")
    main(experiment_config)
