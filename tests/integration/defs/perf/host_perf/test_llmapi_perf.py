# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LLM API-level host performance benchmark.

Bypasses the HTTP serving layer (trtllm-serve + FastAPI + SSE) and calls
LLM.generate() directly. This gives a cleaner host-overhead signal because:
  - No HTTP streaming overhead
  - No FastAPI/SSE framing
  - No client-side parsing overhead

The trade-off is that this does not test the serve path. For host-regression
detection, the signal quality improvement is worth it — HTTP overhead adds
~0.5-1ms of noise per iteration that masks small host regressions.

Design:
- Uses LLM API directly (no trtllm-serve)
- Sends batches of concurrent requests via generate_async()
- Measures wall-clock time for the full batch
- Reports throughput metrics (token/s, request/s)
- Supports YAML config discovery (same configs as E2E tests)

Run:
    LLM_MODELS_ROOT=/path/to/models \
    pytest tests/integration/defs/perf/host_perf/test_llmapi_perf.py -v -s
"""

import glob
import json
import os
import statistics
import time
from typing import Dict, List

import pytest
import yaml

from defs.trt_test_alternative import print_info

from ..open_search_db_utils import (
    add_id,
    generate_perf_yaml,
    get_common_values,
    get_history_data,
    get_job_info,
    post_new_perf_data,
    prepare_baseline_data,
    prepare_regressive_test_cases,
)
from .regression_helper import UPLOAD_TO_DB, get_gpu_type, get_model_dir

HOST_PERF_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Match keys for OpenSearch historical data lookup.
LLMAPI_PERF_MATCH_KEYS = [
    "s_gpu_type",
    "s_runtime",
    "s_model_name",
    "l_tp",
    "l_max_batch_size",
    "l_concurrency",
    "l_isl",
    "l_osl",
]


# ---------------------------------------------------------------------------
# Test case and config classes
# ---------------------------------------------------------------------------


class LlmApiTestCase:
    """A single LLM API benchmark test case parsed from YAML."""

    def __init__(self, yaml_file: str, server_name: str, client_name: str):
        self.yaml_file = yaml_file
        self.server_name = server_name
        self.client_name = client_name
        self._load()

    def _load(self):
        with open(self.yaml_file, "r") as f:
            config = yaml.safe_load(f)

        self.model_name = config.get("metadata", {}).get("model_name", "")

        for sc in config.get("server_configs", []):
            if sc.get("name") == self.server_name:
                self.tp = sc.get("tensor_parallel_size", 1)
                self.max_batch_size = sc.get("max_batch_size", 1)
                self.max_num_tokens = sc.get("max_num_tokens", 256)
                # Build LLM kwargs from server config
                self.llm_kwargs = {}
                for k, v in sc.items():
                    if k in ("name", "model_name", "tensor_parallel_size", "client_configs"):
                        continue
                    self.llm_kwargs[k] = v

                for cc in sc.get("client_configs", []):
                    if cc.get("name") == self.client_name:
                        self.concurrency = cc.get("concurrency", 1)
                        self.iterations = cc.get("iterations", 50)
                        self.isl = cc.get("isl", 128)
                        self.osl = cc.get("osl", 128)
                        return

        raise ValueError(f"Config not found: {self.server_name}/{self.client_name}")

    @property
    def test_id(self) -> str:
        yaml_base = os.path.splitext(os.path.basename(self.yaml_file))[0]
        return f"llmapi-{yaml_base}-{self.server_name}-{self.client_name}"

    def __repr__(self) -> str:
        return self.test_id

    def to_db_data(self) -> dict:
        """Convert test case config to OpenSearch data fields."""
        return {
            "s_server_name": self.server_name,
            "s_model_name": self.model_name.lower(),
            "l_tp": self.tp,
            "l_max_batch_size": self.max_batch_size,
            "l_max_num_tokens": self.max_num_tokens,
            "s_client_name": self.client_name,
            "l_concurrency": self.concurrency,
            "l_iterations": self.iterations,
            "l_isl": self.isl,
            "l_osl": self.osl,
        }


def discover_llmapi_test_cases() -> List[LlmApiTestCase]:
    """Discover test cases from YAML configs."""
    yaml_files = sorted(glob.glob(os.path.join(HOST_PERF_CONFIG_DIR, "host_perf_*.yaml")))
    test_cases = []
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)
        for sc in config.get("server_configs", []):
            server_name = sc.get("name", "")
            for cc in sc.get("client_configs", []):
                client_name = cc.get("name", "")
                test_cases.append(LlmApiTestCase(yaml_file, server_name, client_name))
    return test_cases


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_llmapi_benchmark(
    test_case: LlmApiTestCase,
    output_dir: str,
) -> Dict[str, float]:
    """Run LLM API benchmark directly without HTTP layer.

    Creates an LLM instance, sends concurrent requests, and collects
    throughput metrics.
    """
    from tensorrt_llm.llmapi import LLM, SamplingParams

    model_dir = get_model_dir(test_case.model_name)
    model_path = model_dir if os.path.exists(model_dir) else test_case.model_name

    test_output_dir = os.path.join(output_dir, test_case.test_id)
    os.makedirs(test_output_dir, exist_ok=True)

    # Build LLM with the same config as trtllm-serve would use
    llm = LLM(
        model=model_path,
        backend="pytorch",
        tensor_parallel_size=test_case.tp,
        **test_case.llm_kwargs,
    )

    # Generate synthetic prompts (random token ids)
    prompt_tokens = [1] * test_case.isl
    num_prompts = test_case.concurrency * test_case.iterations

    sampling_params = SamplingParams(
        max_tokens=test_case.osl,
        ignore_eos=True,
    )

    # Warmup: run a small batch first
    warmup_count = min(test_case.concurrency, 4)
    warmup_prompts = [[1] * test_case.isl for _ in range(warmup_count)]
    warmup_results = llm.generate(
        warmup_prompts,
        sampling_params=SamplingParams(max_tokens=8, ignore_eos=True),
    )
    # Wait for warmup to complete
    for r in warmup_results:
        _ = r.outputs

    # Benchmark: send all prompts
    all_prompts = [prompt_tokens for _ in range(num_prompts)]

    wall_start = time.perf_counter()
    results = llm.generate(all_prompts, sampling_params=sampling_params)
    wall_end = time.perf_counter()

    wall_time = wall_end - wall_start

    # Collect per-request metrics
    output_lengths = []
    for r in results:
        for out in r.outputs:
            output_lengths.append(len(out.token_ids))

    total_output_tokens = sum(output_lengths)

    metrics: Dict[str, float] = {
        "wall_time_s": wall_time,
        "num_prompts": float(num_prompts),
        "total_input_tokens": float(num_prompts * test_case.isl),
        "total_output_tokens": float(total_output_tokens),
        "token_throughput": total_output_tokens / wall_time,
        "request_throughput": num_prompts / wall_time,
        "avg_output_len": statistics.mean(output_lengths) if output_lengths else 0.0,
    }

    # Save metrics
    metrics_path = os.path.join(test_output_dir, "llmapi_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Cleanup
    del llm

    return metrics


# ---------------------------------------------------------------------------
# OpenSearch DB integration
# ---------------------------------------------------------------------------


def post_llmapi_perf_to_db(
    results: Dict[str, tuple],
    test_output_dir: str,
):
    """Upload llmapi perf results to OpenSearch DB.

    Follows the same pattern as test_host_perf.py / test_perf_sanity.py.
    Uses s_runtime="llmapi_perf" to distinguish from E2E host_perf data.
    """
    if not results:
        print_info("[llmapi_perf_db] No results to upload.")
        return

    gpu_type = get_gpu_type()
    job_config = get_job_info()
    is_post_merge = job_config["b_is_post_merge"]

    new_data_dict = {}
    cmd_idx = 0

    for test_id, (metrics, test_case) in results.items():
        if not metrics:
            cmd_idx += 1
            continue

        new_data = {
            "s_gpu_type": gpu_type,
            "s_runtime": "llmapi_perf",
        }
        new_data.update(job_config)
        new_data.update(test_case.to_db_data())
        new_data["s_test_case_name"] = test_id

        # Add metrics with d_ prefix
        for metric_name, value in metrics.items():
            new_data[f"d_{metric_name}"] = value

        add_id(new_data)
        new_data_dict[cmd_idx] = new_data
        cmd_idx += 1

    if not new_data_dict:
        print_info("[llmapi_perf_db] No valid data to upload.")
        return

    match_keys = LLMAPI_PERF_MATCH_KEYS

    common_values_dict = get_common_values(new_data_dict, match_keys)

    history_baseline_dict, history_data_dict = get_history_data(
        new_data_dict, match_keys, common_values_dict
    )

    prepare_regressive_test_cases(history_baseline_dict, new_data_dict)

    for cmd_idx, new_data in new_data_dict.items():
        test_name = new_data.get("s_test_case_name", f"cmd_{cmd_idx}")
        if new_data.get("b_is_regression", False):
            print_info(
                f"[llmapi_perf_db] REGRESSION detected for {test_name}: "
                f"{new_data.get('s_regression_info', '')}"
            )
        else:
            print_info(f"[llmapi_perf_db] No regression for {test_name}")

    if is_post_merge and history_baseline_dict is not None:
        new_baseline_data_dict = prepare_baseline_data(
            history_baseline_dict, history_data_dict, new_data_dict
        )
    else:
        new_baseline_data_dict = None

    if UPLOAD_TO_DB:
        post_new_perf_data(new_baseline_data_dict, new_data_dict)

    generate_perf_yaml(new_data_dict, output_dir=test_output_dir)


class LlmapiPerfResultCollector:
    """Session-scoped collector for llmapi perf test results."""

    def __init__(self):
        self.results: Dict[str, tuple] = {}

    def add_result(self, test_id: str, metrics: Dict[str, float], test_case: LlmApiTestCase):
        self.results[test_id] = (metrics, test_case)

    def finalize(self, test_output_dir: str):
        if self.results:
            try:
                post_llmapi_perf_to_db(self.results, test_output_dir)
            except Exception as e:
                print_info(f"[llmapi_perf_db] Failed to upload to OpenSearch: {e}")


_result_collector = LlmapiPerfResultCollector()


@pytest.fixture(scope="session", autouse=True)
def llmapi_perf_db_finalizer(output_dir):
    """Session-scoped fixture that posts accumulated results to OpenSearch DB."""
    yield
    effective_output_dir = output_dir or os.path.join(os.getcwd(), "host_perf_output")
    os.makedirs(effective_output_dir, exist_ok=True)
    _result_collector.finalize(effective_output_dir)


# ---------------------------------------------------------------------------
# Test discovery and parametrization
# ---------------------------------------------------------------------------

LLMAPI_TEST_CASES = discover_llmapi_test_cases()


@pytest.mark.parametrize(
    "llmapi_test_case",
    LLMAPI_TEST_CASES,
    ids=[tc.test_id for tc in LLMAPI_TEST_CASES],
)
def test_llmapi_perf(output_dir, llmapi_test_case):
    """Run LLM API-level host performance benchmark.

    This test bypasses the HTTP serving layer and calls LLM.generate()
    directly. The resulting throughput metrics have less noise from HTTP
    overhead, making them more sensitive to host-side regressions.

    Requires GPU and model weights (set LLM_MODELS_ROOT).
    """
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "host_perf_output")
    os.makedirs(output_dir, exist_ok=True)

    metrics = run_llmapi_benchmark(llmapi_test_case, output_dir)

    assert metrics["total_output_tokens"] > 0, (
        f"No output tokens generated for {llmapi_test_case.test_id}"
    )

    # Store results for session-end DB upload
    _result_collector.add_result(llmapi_test_case.test_id, metrics, llmapi_test_case)

    # Report throughput
    print(f"\nLLMAPI_PERF: {llmapi_test_case.test_id}")
    print(f"  wall_time:          {metrics['wall_time_s']:.2f}s")
    print(f"  num_prompts:        {metrics['num_prompts']}")
    print(f"  token_throughput:   {metrics['token_throughput']:.1f} tok/s")
    print(f"  request_throughput: {metrics['request_throughput']:.2f} req/s")
    print(f"  avg_output_len:     {metrics['avg_output_len']:.1f}")

    # Log key metrics for regression comparison
    for key in ["token_throughput", "request_throughput"]:
        print_info(f"LLMAPI_PERF_METRIC: {llmapi_test_case.test_id} {key}={metrics[key]:.3f}")
