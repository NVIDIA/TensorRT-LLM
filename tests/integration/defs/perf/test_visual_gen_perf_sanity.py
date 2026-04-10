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
"""TensorRT LLM VisualGen perf sanity tests."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_common.error_utils import report_error
from test_common.http_utils import wait_for_endpoint_ready

from defs.trt_test_alternative import print_info
from tensorrt_llm._utils import get_free_port

from ..conftest import get_llm_root, llm_models_root
from .perf_regression_utils import process_and_upload_test_results
from .visual_gen_perf_utils import (
    MAXIMIZE_METRICS,
    MINIMIZE_METRICS,
    REGRESSION_METRICS,
    build_visual_gen_db_entry,
    get_visual_gen_match_keys,
    get_visual_gen_num_gpus_from_server_config,
)

DEFAULT_TIMEOUT = 5400
VISUAL_GEN_CONFIG_FOLDER = os.environ.get(
    "VISUAL_GEN_PERF_SANITY_CONFIG_FOLDER",
    "tests/scripts/perf-sanity/visual_gen",
)
VISUAL_GEN_TEST_TYPES = ["vg_upload", "vg"]
SUPPORTED_GPU_MAPPING = {
    "GB200": "gb200",
    "GB300": "gb300",
    "B200": "b200",
    "B300": "b300",
    "H200": "h200",
}


def to_env_dict(env_vars: str) -> dict[str, str]:
    """Convert a space-separated KEY=VALUE string into an env dict."""
    env = {}
    for env_var in env_vars.split():
        if "=" not in env_var:
            continue
        key, value = env_var.split("=", 1)
        env[key] = value
    return env


def get_gpu_types() -> tuple[str, str]:
    """Return the raw GPU name token and the normalized perf-sanity GPU type."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Failed to get GPU type from nvidia-smi") from exc

    gpu_name = output.splitlines()[0].strip()
    for raw_gpu_type, normalized_gpu_type in SUPPORTED_GPU_MAPPING.items():
        if raw_gpu_type.lower() in gpu_name.lower():
            return raw_gpu_type, normalized_gpu_type

    raise RuntimeError(
        f"Unsupported GPU type for VisualGen perf sanity. nvidia-smi reported: {gpu_name}"
    )


def get_config_dir() -> Path:
    """Return the VisualGen perf sanity config directory."""
    config_dir = Path(VISUAL_GEN_CONFIG_FOLDER)
    if not config_dir.is_absolute():
        config_dir = Path(get_llm_root()) / config_dir
    return config_dir


def _merge_nested_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override values into a base config dict."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_test_case_name(test_case_name: str) -> tuple[str, bool]:
    """Parse the test name into config filename and upload flag."""
    labels = test_case_name.split("-", maxsplit=1)
    if len(labels) != 2 or labels[0] not in VISUAL_GEN_TEST_TYPES:
        raise ValueError(
            "VisualGen perf sanity test name must be one of "
            f"{VISUAL_GEN_TEST_TYPES}: got {test_case_name}"
        )

    config_base_name = labels[1]
    if not config_base_name.endswith((".yaml", ".yml")):
        config_base_name = f"{config_base_name}.yaml"
    return config_base_name, labels[0] == "vg_upload"


def get_visual_gen_test_cases() -> list[str]:
    """Generate all VisualGen perf sanity pytest parameters from the config folder."""
    config_dir = get_config_dir()
    if not config_dir.is_dir():
        return []

    config_names = sorted(
        {path.stem for pattern in ("*.yaml", "*.yml") for path in config_dir.glob(pattern)}
    )
    return [
        f"{test_type}-{config_name}"
        for config_name in config_names
        for test_type in VISUAL_GEN_TEST_TYPES
    ]


class VisualGenPerfSanityTestConfig:
    """Configuration and execution state for one VisualGen perf sanity test case."""

    def __init__(self, test_case_name: str, output_dir: str):
        self._output_dir = output_dir
        self._test_param_labels = test_case_name
        self._benchmark_results: dict[int, dict[str, Any]] = {}
        self.server_log_path = ""
        self.client_log_paths: list[str] = []
        self.test_output_dir = ""

        self.config_file, self.upload_to_db = parse_test_case_name(test_case_name)
        self.raw_gpu_type, self.gpu_type = get_gpu_types()
        self.config_dir = get_config_dir()

        self.case_name = Path(self.config_file).stem
        self.model_name = ""
        self.server_config: dict[str, Any] = {}
        self.client_configs: list[dict[str, Any]] = []
        self.environment: dict[str, Any] = {}
        self.expected_num_gpus = 1
        self.extra_visual_gen_options_path = ""
        self.resolved_extra_visual_gen_options_path = ""
        self.server_recipe_id = ""

    def parse_config_file(self) -> None:
        """Parse the checked-in VisualGen perf sanity YAML."""
        config_path = self.config_dir / self.config_file
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        metadata = config.get("metadata", {})
        hardware = config.get("hardware", {})
        self.environment = config.get("environment", {})
        self.extra_visual_gen_options_path = str(
            config.get("extra_visual_gen_options_path", "") or ""
        )
        self.server_recipe_id = str(config.get("server_recipe_id", "") or "")
        inline_server_config = config.get("server_config") or {}
        self.server_config = self._load_server_config(inline_server_config)
        self.client_configs = config.get("client_configs") or []
        self.model_name = str(metadata.get("model_name", ""))

        if not self.model_name:
            raise ValueError(f"metadata.model_name is required in {config_path}")
        if not self.server_config and not self.extra_visual_gen_options_path:
            raise ValueError(
                f"server_config or extra_visual_gen_options_path is required in {config_path}"
            )
        if not self.client_configs:
            raise ValueError(f"client_configs is required in {config_path}")

        supported_gpus = {str(gpu).upper() for gpu in metadata.get("supported_gpus", [])}
        if supported_gpus and self.raw_gpu_type.upper() not in supported_gpus:
            pytest.skip(
                f"{self.case_name} only supports {sorted(supported_gpus)}, "
                f"but current GPU is {self.raw_gpu_type}"
            )

        self.expected_num_gpus = get_visual_gen_num_gpus_from_server_config(self.server_config)
        gpus_per_node = int(hardware.get("gpus_per_node", self.expected_num_gpus))
        if gpus_per_node != self.expected_num_gpus:
            raise ValueError(
                "hardware.gpus_per_node must match the GPU count derived from "
                f"server_config.parallel: got {gpus_per_node} vs {self.expected_num_gpus}"
            )

    def _resolve_extra_visual_gen_options_path(self) -> str:
        """Resolve the optional external VisualGen config path."""
        config_path = Path(self.extra_visual_gen_options_path)
        if not config_path.is_absolute():
            config_path = Path(get_llm_root()) / config_path
        return str(config_path)

    def _load_server_config(self, inline_server_config: dict[str, Any]) -> dict[str, Any]:
        """Load and merge the optional external VisualGen config with inline overrides."""
        if not self.extra_visual_gen_options_path:
            return inline_server_config

        self.resolved_extra_visual_gen_options_path = self._resolve_extra_visual_gen_options_path()
        with open(self.resolved_extra_visual_gen_options_path, encoding="utf-8") as f:
            extra_config = yaml.safe_load(f) or {}
        if not isinstance(extra_config, dict):
            raise ValueError(
                "extra_visual_gen_options_path must point to a YAML mapping: "
                f"{self.resolved_extra_visual_gen_options_path}"
            )
        return _merge_nested_dicts(extra_config, inline_server_config)

    def _resolve_model_path(self) -> str:
        """Resolve the model path under LLM_MODELS_ROOT, falling back to the original name."""
        raw_model_path = Path(self.model_name)
        if raw_model_path.exists():
            return str(raw_model_path)

        candidate = Path(llm_models_root()) / self.model_name
        if candidate.exists():
            return str(candidate)

        return self.model_name

    def _write_server_config(self) -> str:
        """Materialize server_config into a temporary extra_visual_gen_options YAML."""
        config_path = Path(self.test_output_dir) / "extra_visual_gen_options.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.server_config, f, default_flow_style=False, sort_keys=False)
        return str(config_path)

    def _build_server_command(
        self,
        *,
        model_path: str,
        host: str,
        port: int,
        server_config_path: str,
    ) -> list[str]:
        """Build the trtllm-serve command for VisualGen perf sanity."""
        return [
            "trtllm-serve",
            model_path,
            "--extra_visual_gen_options",
            server_config_path,
            "--host",
            host,
            "--port",
            str(port),
        ]

    @staticmethod
    def _append_optional_arg(
        command: list[str],
        client_config: dict[str, Any],
        key: str,
        flag: str,
    ) -> None:
        """Append an optional CLI arg when the key is present in client_config."""
        value = client_config.get(key)
        if value is None:
            return
        command.extend([flag, str(value)])

    def _build_client_command(
        self,
        *,
        client_config: dict[str, Any],
        host: str,
        port: int,
        server_config_path: str,
        result_dir: str,
        result_filename: str,
    ) -> list[str]:
        """Build the benchmark_visual_gen.py command for one client scenario."""
        command = [
            sys.executable,
            "-m",
            "tensorrt_llm.serve.scripts.benchmark_visual_gen",
            "--model",
            self.model_name,
            "--backend",
            str(client_config["backend"]),
            "--host",
            host,
            "--port",
            str(port),
            "--extra-visual-gen-options",
            server_config_path,
            "--save-result",
            "--result-dir",
            result_dir,
            "--result-filename",
            result_filename,
            "--disable-tqdm",
        ]

        prompt_file = client_config.get("prompt_file")
        if prompt_file is not None:
            prompt_file_path = Path(prompt_file)
            if not prompt_file_path.is_absolute():
                prompt_file_path = Path(get_llm_root()) / prompt_file_path
            command.extend(["--prompt-file", str(prompt_file_path)])
        else:
            prompt = client_config.get("prompt")
            if not prompt:
                raise ValueError(
                    f"client_configs entry in {self.config_file} requires prompt or prompt_file"
                )
            command.extend(["--prompt", str(prompt)])

        optional_arg_map = {
            "num_prompts": "--num-prompts",
            "size": "--size",
            "num_frames": "--num-frames",
            "fps": "--fps",
            "num_inference_steps": "--num-inference-steps",
            "seed": "--seed",
            "max_concurrency": "--max-concurrency",
            "request_rate": "--request-rate",
            "burstiness": "--burstiness",
            "request_timeout": "--request-timeout",
            "guidance_scale": "--guidance-scale",
            "negative_prompt": "--negative-prompt",
            "metric_percentiles": "--metric-percentiles",
        }
        for key, flag in optional_arg_map.items():
            self._append_optional_arg(command, client_config, key, flag)

        extra_body = client_config.get("extra_body")
        if extra_body is not None:
            extra_body_json = extra_body if isinstance(extra_body, str) else json.dumps(extra_body)
            command.extend(["--extra-body", extra_body_json])

        if client_config.get("save_detailed", False):
            command.append("--save-detailed")
        if client_config.get("no_test_input", False):
            command.append("--no-test-input")

        metadata = client_config.get("metadata")
        if metadata:
            metadata_items = [f"{key}={value}" for key, value in metadata.items()]
            command.extend(["--metadata", *metadata_items])

        return command

    def _validate_benchmark_result(
        self,
        *,
        result_data: dict[str, Any],
        client_config: dict[str, Any],
        result_path: str,
    ) -> None:
        """Validate the saved benchmark JSON before it is used as the source of truth."""
        required_keys = [
            "backend",
            "model",
            "total_requests",
            "completed",
            "num_gpus",
            "request_throughput",
            "per_gpu_throughput",
            "mean_e2e_latency_ms",
            "median_e2e_latency_ms",
            "percentiles_e2e_latency_ms",
        ]
        missing_keys = [key for key in required_keys if key not in result_data]
        if missing_keys:
            raise ValueError(f"Missing keys in benchmark result {result_path}: {missing_keys}")

        if str(result_data["backend"]) != str(client_config["backend"]):
            raise ValueError(
                "Benchmark result backend mismatch: "
                f"result={result_data['backend']} config={client_config['backend']}"
            )

        if str(result_data["model"]) != self.model_name:
            raise ValueError(
                f"Benchmark result model mismatch: result={result_data['model']} "
                f"config={self.model_name}"
            )

        if int(result_data["num_gpus"]) != self.expected_num_gpus:
            raise ValueError(
                "Benchmark result GPU count mismatch: "
                f"result={result_data['num_gpus']} expected={self.expected_num_gpus}"
            )

        total_requests = int(result_data["total_requests"])
        completed_requests = int(result_data["completed"])
        if completed_requests != total_requests:
            raise RuntimeError(
                "Benchmark result contains failed requests: "
                f"completed={completed_requests}, total={total_requests}"
            )

        percentiles = result_data.get("percentiles_e2e_latency_ms", {})
        for percentile in ("p90", "p99"):
            if percentile not in percentiles:
                raise ValueError(
                    f"Missing {percentile} E2E latency in benchmark result {result_path}"
                )

    def _load_benchmark_result(
        self,
        *,
        result_path: str,
        client_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Load and validate one saved benchmark JSON."""
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Benchmark result JSON not found: {result_path}")

        with open(result_path, encoding="utf-8") as f:
            result_data = json.load(f)

        self._validate_benchmark_result(
            result_data=result_data,
            client_config=client_config,
            result_path=result_path,
        )
        return result_data

    def run(self) -> None:
        """Run the VisualGen perf sanity server and all benchmark clients."""
        self.test_output_dir = os.path.join(self._output_dir, self._test_param_labels)
        os.makedirs(self.test_output_dir, exist_ok=True)
        result_dir = os.path.join(self.test_output_dir, "benchmark_results")
        os.makedirs(result_dir, exist_ok=True)

        server_host = "localhost"
        server_port = get_free_port()
        model_path = self._resolve_model_path()
        server_config_path = self._write_server_config()
        self.server_log_path = os.path.join(self.test_output_dir, "trtllm-serve.log")

        server_command = self._build_server_command(
            model_path=model_path,
            host=server_host,
            port=server_port,
            server_config_path=server_config_path,
        )
        server_env = os.environ.copy()
        server_env.update(to_env_dict(str(self.environment.get("server_env_var", ""))))

        server_proc = None
        try:
            print_info(f"Starting VisualGen perf sanity server: {server_command}")
            with open(self.server_log_path, "w", encoding="utf-8") as server_log:
                server_proc = subprocess.Popen(
                    server_command,
                    env=server_env,
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                )

            wait_for_endpoint_ready(
                f"http://{server_host}:{server_port}/health",
                timeout=DEFAULT_TIMEOUT,
                check_files=[self.server_log_path],
                server_proc=server_proc,
            )

            client_env = os.environ.copy()
            client_env.update(to_env_dict(str(self.environment.get("client_env_var", ""))))
            for client_idx, client_config in enumerate(self.client_configs):
                result_filename = f"visual_gen_benchmark.{client_idx}.json"
                result_path = os.path.join(result_dir, result_filename)
                client_log_path = os.path.join(
                    self.test_output_dir,
                    f"benchmark_visual_gen.{client_idx}.log",
                )
                self.client_log_paths.append(client_log_path)

                client_command = self._build_client_command(
                    client_config=client_config,
                    host=server_host,
                    port=server_port,
                    server_config_path=server_config_path,
                    result_dir=result_dir,
                    result_filename=result_filename,
                )
                print_info(f"Running VisualGen benchmark client: {client_command}")
                result = subprocess.run(
                    client_command,
                    env=client_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                with open(client_log_path, "w", encoding="utf-8") as client_log:
                    client_log.write(result.stdout)

                if result.returncode != 0:
                    raise RuntimeError(
                        f"benchmark_visual_gen.py failed with exit code {result.returncode}"
                    )

                self._benchmark_results[client_idx] = self._load_benchmark_result(
                    result_path=result_path,
                    client_config=client_config,
                )
        finally:
            if server_proc is not None:
                server_proc.terminate()
                try:
                    server_proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server_proc.kill()
                    server_proc.wait(timeout=30)

    def get_log_files(self) -> list[str]:
        """Return the server and client logs collected during the test."""
        log_files = []
        if self.server_log_path:
            log_files.append(self.server_log_path)
        log_files.extend(self.client_log_paths)
        return log_files

    def upload_test_results_to_database(self) -> None:
        """Build OpenSearch documents and run the shared perf regression/upload pipeline."""
        if len(self._benchmark_results) != len(self.client_configs):
            raise RuntimeError(
                "VisualGen benchmark result count mismatch: "
                f"results={len(self._benchmark_results)}, clients={len(self.client_configs)}"
            )

        new_data_dict = {}
        for client_idx, client_config in enumerate(self.client_configs):
            result_data = self._benchmark_results.get(client_idx)
            if result_data is None:
                raise RuntimeError(f"Missing benchmark result for client index {client_idx}")

            new_data_dict[client_idx] = build_visual_gen_db_entry(
                gpu_type=self.gpu_type,
                model_name=self.model_name,
                server_config=self.server_config,
                client_config=client_config,
                result_data=result_data,
                case_name=self.case_name,
                extra_visual_gen_options_path=self.extra_visual_gen_options_path,
                server_recipe_id=self.server_recipe_id,
            )

        extra_fields = {
            "s_stage_name": os.environ.get("stageName", ""),
            "s_test_list": self._test_param_labels,
        }
        process_and_upload_test_results(
            new_data_dict=new_data_dict,
            match_keys=get_visual_gen_match_keys(self.server_recipe_id),
            maximize_metrics=MAXIMIZE_METRICS,
            minimize_metrics=MINIMIZE_METRICS,
            regression_metrics=REGRESSION_METRICS,
            extra_fields=extra_fields,
            upload_to_db=self.upload_to_db,
        )


VISUAL_GEN_PERF_SANITY_TEST_CASES = get_visual_gen_test_cases()


@pytest.mark.parametrize("visual_gen_perf_sanity_test_case", VISUAL_GEN_PERF_SANITY_TEST_CASES)
def test_visual_gen_e2e(output_dir, visual_gen_perf_sanity_test_case):
    """Run one VisualGen perf sanity case end-to-end."""
    config = VisualGenPerfSanityTestConfig(visual_gen_perf_sanity_test_case, output_dir)

    try:
        config.parse_config_file()
        config.run()
        config.upload_test_results_to_database()
    except Exception as exc:
        report_error(error_msg=exc, log_files=config.get_log_files())
