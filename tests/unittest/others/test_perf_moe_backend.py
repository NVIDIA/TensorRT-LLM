# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__extra_import_path__ = ["../../integration"]

import pathlib

import pytest
import yaml
from defs.perf import test_perf as perf_test

_CASE = (
    "deepseek_v4_pro_nvfp4_dspark-bench-pytorch-float4-maxbs:64-maxnt:9216-"
    "kv_frac:0.5-input_output_len:8192,1024-reqs:32-con:8-ep:8-gpus:8"
)


@pytest.mark.parametrize("backend", [None, "AUTO", "TRTLLM", "CUTEDSL"])
def test_moe_backend_round_trip(monkeypatch: pytest.MonkeyPatch, backend: str | None) -> None:
    monkeypatch.setenv("TRTLLM_TOTAL_GPU_COUNT", "8")
    monkeypatch.setattr(perf_test.PerfTestConfig, "get_benchmark_type", lambda self: "gpt")
    case = _CASE if backend is None else f"{_CASE}-moe:{backend}"
    config = perf_test.PerfTestConfig()
    config.load_from_str(case)
    assert config.moe_backend == backend
    assert config.to_string() == case
    # Parsing another case must clear the previous override.
    config.load_from_str(_CASE)
    assert config.moe_backend is None


@pytest.mark.parametrize("backend", ["AUTO", "TRTLLM", "CUTEDSL"])
def test_nvfp4_dspark_bench_override(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    monkeypatch.setenv("TRTLLM_TOTAL_GPU_COUNT", "8")
    monkeypatch.setattr(perf_test.PerfTestConfig, "get_benchmark_type", lambda self: "gpt")
    monkeypatch.setattr(perf_test, "get_model_dir", lambda name: "/models/dspark")
    runner = object.__new__(perf_test.MultiMetricPerfTest)
    runner._config = perf_test.PerfTestConfig()
    runner._config.load_from_str(f"{_CASE}-moe:{backend}")
    runner._benchmark_script = "trtllm-bench"
    runner.lora_dirs = []

    command = runner.get_trtllm_bench_command(str(tmp_path))
    config_path = next(
        arg.removeprefix("--config=") for arg in command if arg.startswith("--config=")
    )
    config = yaml.safe_load(pathlib.Path(config_path).read_text())
    baseline = perf_test.get_model_yaml_config(_CASE)
    assert config["moe_config"].pop("backend") == backend
    baseline["moe_config"].pop("backend")
    config["kv_cache_config"].pop("avg_seq_len", None)
    assert config == baseline
    assert config["speculative_config"]["decoding_type"] == "DSpark"
    assert config["speculative_config"]["block_size"] == 3
    assert config["speculative_config"]["speculative_model"].endswith(
        "DeepSeek-V4-Pro-nvfp4-DSpark"
    )
    assert "--custom_tokenizer=deepseek_v4" in command
    assert "--tp=8" in command
    assert "--ep=8" in command


def test_empty_moe_backend_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRTLLM_TOTAL_GPU_COUNT", "8")
    with pytest.raises(AssertionError, match="moe backend must not be empty"):
        perf_test.PerfTestConfig().load_from_str(f"{_CASE}-moe:")
