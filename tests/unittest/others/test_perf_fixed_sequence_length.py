# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "tests" / "integration"))

from defs.perf import test_perf as perf_test  # noqa: E402


@pytest.mark.parametrize(
    ("input_lens", "output_lens", "expected"),
    [
        ([500], [2000], 2500),
        ([500, 1000], [2000, 1500], 2500),
        ([500, 1000], [2000, 2000], None),
    ],
)
def test_fixed_dataset_sequence_length(input_lens, output_lens, expected):
    config = perf_test.PerfTestConfig(
        runtime="bench",
        input_lens=input_lens,
        output_lens=output_lens,
    )

    assert config.get_fixed_dataset_sequence_length() == expected


@pytest.mark.parametrize(
    ("runtime", "model_name", "num_loras", "build_only"),
    [
        ("bench", "", 0, True),
        ("bench", "", 1, False),
        ("serve", "qwen3_4b_eagle3", 0, False),
        ("serve", "nemotron_3_nano_omni_nvfp4_image", 0, False),
    ],
)
def test_variable_dataset_does_not_infer_sequence_length(
    runtime,
    model_name,
    num_loras,
    build_only,
):
    config = perf_test.PerfTestConfig(
        model_name=model_name,
        runtime=runtime,
        input_lens=[500],
        output_lens=[2000],
        num_loras=num_loras,
    )
    config.build_only = build_only

    assert config.get_fixed_dataset_sequence_length() is None


@pytest.mark.parametrize(
    ("runtime", "backend", "base_config", "expected"),
    [
        ("bench", "pytorch", {}, 2500),
        ("serve", "", {}, 2500),
        ("bench", "", {}, None),
        ("bench", "_autodeploy", {}, None),
        ("bench", "pytorch", {"kv_cache_config": None}, 2500),
        ("bench", "pytorch", {"max_seq_len": 2048}, None),
    ],
)
def test_model_yaml_infers_avg_seq_len_for_supported_configs(
    monkeypatch,
    runtime,
    backend,
    base_config,
    expected,
):
    runner = object.__new__(perf_test.MultiMetricPerfTest)
    runner._config = perf_test.PerfTestConfig(
        runtime=runtime,
        backend=backend,
        input_lens=[500],
        output_lens=[2000],
    )
    runner.lora_dirs = []
    monkeypatch.setattr(
        perf_test,
        "get_model_yaml_config",
        lambda *args, **kwargs: base_config.copy(),
    )

    config = runner._get_model_yaml_config()

    kv_cache_config = config.get("kv_cache_config") or {}
    assert kv_cache_config.get("avg_seq_len") == expected


def test_model_yaml_preserves_explicit_avg_seq_len(monkeypatch):
    runner = object.__new__(perf_test.MultiMetricPerfTest)
    runner._config = perf_test.PerfTestConfig(
        runtime="bench",
        backend="pytorch",
        input_lens=[500],
        output_lens=[2000],
    )
    runner.lora_dirs = []
    monkeypatch.setattr(
        perf_test,
        "get_model_yaml_config",
        lambda *args, **kwargs: {"kv_cache_config": {"avg_seq_len": 1024}},
    )

    config = runner._get_model_yaml_config()

    assert config["kv_cache_config"]["avg_seq_len"] == 1024


def test_model_yaml_skips_avg_seq_len_for_older_schema(monkeypatch):
    from tensorrt_llm.llmapi import llm_args

    class LegacyKvCacheConfig:
        model_fields = {}

    runner = object.__new__(perf_test.MultiMetricPerfTest)
    runner._config = perf_test.PerfTestConfig(
        runtime="bench",
        backend="pytorch",
        input_lens=[500],
        output_lens=[2000],
    )
    runner.lora_dirs = []
    monkeypatch.setattr(perf_test, "get_model_yaml_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(llm_args, "KvCacheConfig", LegacyKvCacheConfig)

    config = runner._get_model_yaml_config()

    assert "kv_cache_config" not in config
