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
"""The perf-sanity warmup pass must send the measured traffic shape and nothing else.

A pass whose command drifts from the measured one warms the wrong prefill shapes, and
one that reuses the measured prompts lets measured requests hit its KV cache blocks;
either way the measured numbers move without any runtime change.
"""

import importlib.util
from pathlib import Path
from typing import List, Sequence

import pytest

pytestmark = pytest.mark.cpu_only

# Loaded by path: importing it through the ``defs`` package pulls in torch.
_SPEC = importlib.util.spec_from_file_location(
    "warmup_pass",
    Path(__file__).resolve().parents[2] / "integration/defs/perf/warmup_pass.py",
)
warmup_pass = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(warmup_pass)

_BENCHMARK_SERVING = ["python", "-m", "tensorrt_llm.serve.scripts.benchmark_serving"]

# Shaped like ClientConfig._to_default_benchmark_cmd's output for the three modes.
# fmt: off
_AGGR_RANDOM_CMD = _BENCHMARK_SERVING + [
    "--model", "/models/m", "--tokenizer", "/models/m",
    "--num-prompts", "40", "--max-concurrency", "4",
    "--percentile-metrics", "ttft,tpot,itl,e2el", "--ignore-eos", "--no-test-input",
    "--dataset-name", "random", "--random-ids", "--tokenize-on-client",
    "--random-input-len", "8192", "--random-output-len", "1024",
    "--random-range-ratio", "0.0", "--backend", "openai", "--trust-remote-code",
]
_E2E_DATASET_TIME_BREAKDOWN_CMD = _BENCHMARK_SERVING + [
    "--model", "/models/m", "--tokenizer", "/models/m",
    "--num-prompts", "43010", "--max-concurrency", "4301",
    "--percentile-metrics", "ttft,tpot,itl,e2el", "--ignore-eos",
    "--dataset-name", "trtllm_custom", "--dataset-path", "/datasets/d.json",
    "--backend", "openai", "--trust-remote-code",
    "--save-request-time-breakdown", "/out/time_breakdown",
]
_CTX_ONLY_CMD = _BENCHMARK_SERVING + [
    "--model", "/models/m", "--tokenizer", "/models/m",
    "--num-prompts", "6660", "--max-concurrency", "666",
    "--percentile-metrics", "ttft,tpot,itl,e2el", "--ignore-eos",
    "--dataset-name", "random", "--random-ids", "--tokenize-on-client",
    "--random-input-len", "7386", "--random-output-len", "1",
    "--random-range-ratio", "0.0", "--backend", "openai", "--non-streaming",
]
# fmt: on

_REPLACED_FLAGS_WITH_VALUE = ("--num-prompts", "--seed", "--save-request-time-breakdown")


def _value_of(cmd: Sequence[str], flag: str) -> str:
    assert cmd.count(flag) == 1, f"{flag} appears {cmd.count(flag)} times in {cmd}"
    return cmd[cmd.index(flag) + 1]


def _traffic_shape(cmd: Sequence[str]) -> List[str]:
    """The command without the flags the warmup pass may change."""
    shape, args = [], iter(cmd)
    for arg in args:
        if arg in _REPLACED_FLAGS_WITH_VALUE:
            next(args, None)
        elif arg != "--no-test-input":
            shape.append(arg)
    return shape


@pytest.mark.parametrize(
    "measured_cmd", [_AGGR_RANDOM_CMD, _E2E_DATASET_TIME_BREAKDOWN_CMD, _CTX_ONLY_CMD]
)
def test_warmup_pass_cmd_keeps_the_measured_traffic_shape(measured_cmd: List[str]) -> None:
    concurrency = int(_value_of(measured_cmd, "--max-concurrency"))
    warmup_cmd = warmup_pass.build_warmup_pass_cmd(measured_cmd, concurrency)

    assert _traffic_shape(warmup_cmd) == _traffic_shape(measured_cmd)
    assert _value_of(warmup_cmd, "--num-prompts") == str(concurrency)
    assert warmup_cmd.count("--no-test-input") == 1
    assert "--save-request-time-breakdown" not in warmup_cmd
    assert "/out/time_breakdown" not in warmup_cmd


@pytest.mark.parametrize(
    ("seed_args", "expected_warmup_seed"),
    [
        # benchmark_serving's own default seed is 0.
        ([], "1"),
        (["--seed", "7"], "8"),
    ],
)
def test_warmup_pass_draws_other_prompts_than_the_measured_client(
    seed_args: List[str], expected_warmup_seed: str
) -> None:
    measured_cmd = _AGGR_RANDOM_CMD + seed_args
    warmup_cmd = warmup_pass.build_warmup_pass_cmd(measured_cmd, 4)

    assert _value_of(warmup_cmd, "--seed") == expected_warmup_seed


def test_warmup_pass_cmd_leaves_the_measured_cmd_untouched() -> None:
    measured_cmd = list(_E2E_DATASET_TIME_BREAKDOWN_CMD)
    warmup_pass.build_warmup_pass_cmd(measured_cmd, 4301)

    assert measured_cmd == _E2E_DATASET_TIME_BREAKDOWN_CMD


@pytest.mark.parametrize("num_requests", [0, -1])
def test_warmup_pass_needs_a_request(num_requests: int) -> None:
    with pytest.raises(ValueError):
        warmup_pass.build_warmup_pass_cmd(_AGGR_RANDOM_CMD, num_requests)


@pytest.mark.parametrize(
    ("environ", "enabled"),
    [
        ({}, True),
        ({warmup_pass.WARMUP_PASS_ENV: "1"}, True),
        ({warmup_pass.WARMUP_PASS_ENV: "0"}, False),
        ({warmup_pass.WARMUP_PASS_ENV: " False "}, False),
        ({warmup_pass.WARMUP_PASS_ENV: "OFF"}, False),
        ({warmup_pass.WARMUP_PASS_ENV: "no"}, False),
    ],
)
def test_warmup_pass_env_switch(environ: dict, enabled: bool) -> None:
    assert warmup_pass.warmup_pass_enabled(environ) is enabled


@pytest.mark.parametrize(
    ("benchmark_mode", "applies"),
    [
        (None, True),
        ("", True),
        ("ctx_only", True),
        ("e2e", True),
        ("gen_only", False),
        ("gen_only_no_context", False),
    ],
)
def test_warmup_pass_modes(benchmark_mode, applies: bool) -> None:
    assert warmup_pass.warmup_pass_applies(benchmark_mode) is applies
