from pathlib import Path
from unittest.mock import patch

import yaml

from tensorrt_llm.configure.cli import BenchmarkSubCommand, TRTLLMConfigure
from tensorrt_llm.configure.profile import InferenceMaxProfile


def test_trtllm_configure_subcommand_basic(tmp_path: Path):
    output_path = tmp_path / "test_config.yaml"

    mock_config = {
        "max_batch_size": 1,
        "kv_cache_free_gpu_memory_fraction": 0.9,
    }

    cmd = BenchmarkSubCommand(
        model="meta-llama/Llama-3.1-8B",
        gpu="H200_SXM",
        num_gpus=1,
        isl=1000,
        osl=2000,
        concurrency=64,
        tp_size=1,
        output=output_path,
        profile="inferencemax",
    )

    trtllm_configure = TRTLLMConfigure(benchmark=cmd)

    # Mock get_config to return our mock config
    with patch.object(InferenceMaxProfile, "get_config", return_value=mock_config):
        trtllm_configure.run()

    assert output_path.exists()
    with open(output_path, "r") as f:
        loaded_config = yaml.safe_load(f)

    assert "max_batch_size" in loaded_config
    assert loaded_config["max_batch_size"] == 1
    assert "kv_cache_free_gpu_memory_fraction" in loaded_config
    assert loaded_config["kv_cache_free_gpu_memory_fraction"] == 0.9
