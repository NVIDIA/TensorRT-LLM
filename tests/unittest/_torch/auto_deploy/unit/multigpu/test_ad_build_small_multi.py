"""Testing build_and_run_ad end2end."""

from typing import Dict

import pytest
from _model_test_utils import get_small_model_config_pytest_param
from build_and_run_ad import ExperimentConfig, main


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("mode", ["graph", "transformers"])
@pytest.mark.parametrize(
    "experiment_config, attn_backend, compile_backend",
    [
        get_small_model_config_pytest_param(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            attn_backend="flashinfer",
            compile_backend="torch-opt",
        ),
    ],
)
def test_build_ad(
    world_size: int, experiment_config: Dict, attn_backend: str, compile_backend: str, mode: str
):
    experiment_config["args"]["world_size"] = world_size
    experiment_config["args"]["runtime"] = "trtllm"  # Default runtime set to trtllm
    experiment_config["args"]["mode"] = mode
    experiment_config["args"]["transforms"] = (
        {
            "resize_kv_cache": {
                "stage": "cache_init",
                "free_mem_ratio": 0.00,
            },
            "match_attention_layout": {
                "stage": "pattern_matcher",
                "attn_backend": attn_backend,
            },
            "insert_cached_attention": {
                "stage": "cache_init",
                "attn_backend": attn_backend,
            },
            "compile_model": {
                "stage": "compile",
                "compile_backend": compile_backend,
            },
        }
        if mode == "graph"
        else {
            "transformers_replace_cached_attn": {
                "stage": "cache_init",
                "attn_backend": attn_backend,
            },
        }
    )
    experiment_config = ExperimentConfig(**experiment_config)
    print(f"Experiment Config: {experiment_config}")
    main(experiment_config)
