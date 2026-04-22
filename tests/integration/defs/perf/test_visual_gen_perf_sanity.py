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


def parse_select_pattern(select_pattern: str) -> list[str]:
    """Parse one or more server_config names."""
    return [name.strip() for name in select_pattern.split(",") if name.strip()]


def parse_test_case_name(test_case_name: str) -> tuple[str, str | None, bool]:
    """Parse the test name into config filename, server selector, and upload flag."""
    labels = test_case_name.split("-")
    if len(labels) < 2 or labels[0] not in VISUAL_GEN_TEST_TYPES:
        raise ValueError(
            "VisualGen perf sanity test name must be one of "
            f"{VISUAL_GEN_TEST_TYPES}: got {test_case_name}"
        )

    config_base_name = labels[1]
    if not config_base_name.endswith((".yaml", ".yml")):
        config_base_name = f"{config_base_name}.yaml"
    select_pattern = "-".join(labels[2:]) if len(labels) > 2 else None
    return config_base_name, select_pattern, labels[0] == "vg_upload"


def get_server_config_names(yaml_path: Path) -> list[str]:
    """Read a VisualGen perf YAML and return all named server configs."""
    try:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (FileNotFoundError, OSError, yaml.YAMLError):
        return []

    server_configs = data.get("server_configs") or []
    if not isinstance(server_configs, list):
        return []

    return [str(config.get("name", "")) for config in server_configs if config.get("name")]


def get_visual_gen_test_cases() -> list[str]:
    """Generate VisualGen perf sanity pytest parameters from the config folder."""
    config_dir = get_config_dir()
    if not config_dir.is_dir():
        return []

    config_paths = sorted(
        {path for pattern in ("*.yaml", "*.yml") for path in config_dir.glob(pattern)}
    )
    test_cases = []
    for config_path in config_paths:
        config_name = config_path.stem
        server_names = get_server_config_names(config_path)
        for test_type in VISUAL_GEN_TEST_TYPES:
            test_cases.append(f"{test_type}-{config_name}")
            for server_name in server_names:
                test_cases.append(f"{test_type}-{config_name}-{server_name}")
    return test_cases


class VisualGenPerfSanityTestConfig:
    """Configuration and execution state for one VisualGen perf sanity test case."""

    def __init__(self, test_case_name: str, output_dir: str):
        self._output_dir = output_dir
        self._test_param_labels = test_case_name
        self._benchmark_results: dict[int, dict[int, dict[str, Any]]] = {}
        self.log_files: list[str] = []
        self.test_output_dir = ""

        self.config_file, self.select_pattern, self.upload_to_db = parse_test_case_name(
            test_case_name
        )
        self.raw_gpu_type, self.gpu_type = get_gpu_types()
        self.config_dir = get_config_dir()

        self.case_name = Path(self.config_file).stem
        self.default_model_name = ""
        self.server_cases: list[dict[str, Any]] = []
        self.environment: dict[str, Any] = {}

    def parse_config_file(self) -> None:
        """Parse the checked-in VisualGen perf sanity YAML."""
        config_path = self.config_dir / self.config_file
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        metadata = config.get("metadata", {})
        hardware = config.get("hardware", {})
        self.environment = config.get("environment", {})
        self.default_model_name = str(metadata.get("model_name", ""))
        shared_extra_visual_gen_options_path = str(
            config.get("extra_visual_gen_options_path", "") or ""
        )

        if not self.default_model_name:
            raise ValueError(f"metadata.model_name is required in {config_path}")

        supported_gpus = {str(gpu).upper() for gpu in metadata.get("supported_gpus", [])}
        if supported_gpus and self.raw_gpu_type.upper() not in supported_gpus:
            pytest.skip(
                f"{self.case_name} only supports {sorted(supported_gpus)}, "
                f"but current GPU is {self.raw_gpu_type}"
            )

        raw_server_configs = config.get("server_configs")
        if raw_server_configs is None:
            raise ValueError(f"server_configs is required in {config_path}")
        if not isinstance(raw_server_configs, list):
            raise ValueError(f"server_configs must be a list in {config_path}")

        selected_server_names = (
            parse_select_pattern(self.select_pattern) if self.select_pattern else None
        )
        server_cases = []
        for server_config_data in raw_server_configs:
            if not isinstance(server_config_data, dict):
                raise ValueError(f"Each server_configs entry must be a mapping in {config_path}")

            server_name = str(server_config_data.get("name", "") or "")
            if not server_name:
                raise ValueError(f"server_configs[].name is required in {config_path}")
            if selected_server_names is not None and server_name not in selected_server_names:
                continue

            model_name = str(server_config_data.get("model_name") or self.default_model_name)
            model_path = str(server_config_data.get("model_path") or model_name)
            extra_visual_gen_options_path = str(
                server_config_data.get("extra_visual_gen_options_path")
                or shared_extra_visual_gen_options_path
                or ""
            )
            inline_server_config = server_config_data.get("server_config") or {}
            server_config, resolved_extra_visual_gen_options_path = self._load_server_config(
                extra_visual_gen_options_path, inline_server_config
            )
            server_config = self._resolve_server_config_paths(server_config)
            client_configs = server_config_data.get("client_configs") or []

            if not server_config and not extra_visual_gen_options_path:
                raise ValueError(
                    "server_config or extra_visual_gen_options_path is required in "
                    f"{config_path} for {server_name}"
                )
            if not client_configs:
                raise ValueError(f"client_configs is required in {config_path} for {server_name}")

            expected_num_gpus = get_visual_gen_num_gpus_from_server_config(server_config)
            gpus_per_node = int(hardware.get("gpus_per_node", expected_num_gpus))
            if gpus_per_node != expected_num_gpus:
                raise ValueError(
                    "hardware.gpus_per_node must match the GPU count derived from "
                    f"server_config.parallel: got {gpus_per_node} vs {expected_num_gpus} "
                    f"for {server_name}"
                )

            server_cases.append(
                {
                    "name": server_name,
                    "model_name": model_name,
                    "model_path": model_path,
                    "server_config": server_config,
                    "client_configs": client_configs,
                    "expected_num_gpus": expected_num_gpus,
                    "extra_visual_gen_options_path": extra_visual_gen_options_path,
                    "resolved_extra_visual_gen_options_path": resolved_extra_visual_gen_options_path,
                }
            )

        if not server_cases:
            if selected_server_names is None:
                raise ValueError(f"No server_configs found in {config_path}")
            raise ValueError(f"No server_configs matched {selected_server_names} in {config_path}")

        self.server_cases = server_cases

    def _resolve_extra_visual_gen_options_path(self, path_value: str) -> str:
        """Resolve the optional external VisualGen config path."""
        config_path = Path(path_value)
        if not config_path.is_absolute():
            config_path = Path(get_llm_root()) / config_path
        return str(config_path)

    def _resolve_asset_path(self, path_value: str) -> str:
        """Resolve a model or asset path against LLM_MODELS_ROOT when possible."""
        if not path_value:
            return path_value

        models_root = Path(llm_models_root())
        raw_path = Path(path_value)
        candidates: list[Path] = []

        def _append_candidate(candidate: Path) -> None:
            if candidate not in candidates:
                candidates.append(candidate)

        _append_candidate(raw_path)
        if raw_path.is_absolute():
            raw_path_str = str(raw_path)
            if "/common/cache/hub/" in raw_path_str:
                suffix = raw_path_str.split("/common/cache/hub/", 1)[1]
                _append_candidate(models_root / suffix)
            if "/common/cache/" in raw_path_str:
                suffix = raw_path_str.split("/common/cache/", 1)[1]
                _append_candidate(models_root / suffix)
            _append_candidate(models_root / raw_path.name)
        else:
            _append_candidate(models_root / path_value)
            _append_candidate(models_root / raw_path.name)
            if "/" in path_value:
                _append_candidate(models_root / path_value.split("/", 1)[1])

        if "gemma-3-12b-it" in path_value:
            _append_candidate(models_root / "gemma" / "gemma-3-12b-it")
            _append_candidate(models_root / "gemma-3-12b-it")

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        raise FileNotFoundError(
            f"Asset not found for {path_value!r}. Tried: {[str(c) for c in candidates]}"
        )

    def _resolve_server_config_paths(self, server_config: dict[str, Any]) -> dict[str, Any]:
        """Resolve model asset paths embedded in the serve config."""
        resolved_config = copy.deepcopy(server_config)
        for key in (
            "checkpoint_path",
            "text_encoder_path",
            "spatial_upsampler_path",
            "distilled_lora_path",
        ):
            value = resolved_config.get(key)
            if value:
                resolved_config[key] = self._resolve_asset_path(str(value))
        return resolved_config

    def _load_server_config(
        self,
        extra_visual_gen_options_path: str,
        inline_server_config: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        """Load and merge the optional external VisualGen config with inline overrides."""
        if not extra_visual_gen_options_path:
            return inline_server_config, ""

        resolved_path = self._resolve_extra_visual_gen_options_path(extra_visual_gen_options_path)
        with open(resolved_path, encoding="utf-8") as f:
            extra_config = yaml.safe_load(f) or {}
        if not isinstance(extra_config, dict):
            raise ValueError(
                f"extra_visual_gen_options_path must point to a YAML mapping: {resolved_path}"
            )
        return _merge_nested_dicts(extra_config, inline_server_config), resolved_path

    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve the configured model path under LLM_MODELS_ROOT when possible."""
        return self._resolve_asset_path(model_path)

    def _write_server_config(self, output_dir: str, server_config: dict[str, Any]) -> str:
        """Materialize server_config into a temporary extra_visual_gen_options YAML."""
        config_path = Path(output_dir) / "extra_visual_gen_options.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(server_config, f, default_flow_style=False, sort_keys=False)
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
        model_path: str,
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
            model_path,
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
        model_path: str,
        expected_num_gpus: int,
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

        if str(result_data["model"]) != model_path:
            raise ValueError(
                f"Benchmark result model mismatch: result={result_data['model']} "
                f"config={model_path}"
            )

        if int(result_data["num_gpus"]) != expected_num_gpus:
            raise ValueError(
                "Benchmark result GPU count mismatch: "
                f"result={result_data['num_gpus']} expected={expected_num_gpus}"
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
        model_path: str,
        expected_num_gpus: int,
        result_path: str,
        client_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Load and validate one saved benchmark JSON."""
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Benchmark result JSON not found: {result_path}")

        with open(result_path, encoding="utf-8") as f:
            result_data = json.load(f)

        self._validate_benchmark_result(
            model_path=model_path,
            expected_num_gpus=expected_num_gpus,
            result_data=result_data,
            client_config=client_config,
            result_path=result_path,
        )
        return result_data

    def run(self) -> None:
        """Run the VisualGen perf sanity server and all benchmark clients."""
        self.test_output_dir = os.path.join(self._output_dir, self._test_param_labels)
        os.makedirs(self.test_output_dir, exist_ok=True)
        server_env = os.environ.copy()
        server_env.update(to_env_dict(str(self.environment.get("server_env_var", ""))))
        client_env = os.environ.copy()
        client_env.update(to_env_dict(str(self.environment.get("client_env_var", ""))))
        server_host = "localhost"

        for server_idx, server_case in enumerate(self.server_cases):
            server_name = str(server_case["name"])
            safe_server_name = server_name.replace(os.sep, "_")
            server_output_dir = os.path.join(
                self.test_output_dir, f"{server_idx:02d}_{safe_server_name}"
            )
            os.makedirs(server_output_dir, exist_ok=True)
            result_dir = os.path.join(server_output_dir, "benchmark_results")
            os.makedirs(result_dir, exist_ok=True)

            model_path = str(server_case["model_path"])
            expected_num_gpus = int(server_case["expected_num_gpus"])
            server_port = get_free_port()
            resolved_model_path = self._resolve_model_path(model_path)
            server_config_path = self._write_server_config(
                server_output_dir, server_case["server_config"]
            )
            server_log_path = os.path.join(server_output_dir, "trtllm-serve.log")
            self.log_files.append(server_log_path)

            server_command = self._build_server_command(
                model_path=resolved_model_path,
                host=server_host,
                port=server_port,
                server_config_path=server_config_path,
            )

            server_proc = None
            try:
                print_info(
                    f"Starting VisualGen perf sanity server for {server_name}: {server_command}"
                )
                with open(server_log_path, "w", encoding="utf-8") as server_log:
                    server_proc = subprocess.Popen(
                        server_command,
                        env=server_env,
                        stdout=server_log,
                        stderr=subprocess.STDOUT,
                    )

                wait_for_endpoint_ready(
                    f"http://{server_host}:{server_port}/health",
                    timeout=DEFAULT_TIMEOUT,
                    check_files=[server_log_path],
                    server_proc=server_proc,
                )

                server_results: dict[int, dict[str, Any]] = {}
                for client_idx, client_config in enumerate(server_case["client_configs"]):
                    result_filename = f"visual_gen_benchmark.{client_idx}.json"
                    result_path = os.path.join(result_dir, result_filename)
                    client_log_path = os.path.join(
                        server_output_dir,
                        f"benchmark_visual_gen.{client_idx}.log",
                    )
                    self.log_files.append(client_log_path)

                    client_command = self._build_client_command(
                        model_path=resolved_model_path,
                        client_config=client_config,
                        host=server_host,
                        port=server_port,
                        server_config_path=server_config_path,
                        result_dir=result_dir,
                        result_filename=result_filename,
                    )
                    print_info(
                        f"Running VisualGen benchmark client for {server_name}: {client_command}"
                    )
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
                            "benchmark_visual_gen.py failed with exit code "
                            f"{result.returncode} for {server_name}"
                        )

                    server_results[client_idx] = self._load_benchmark_result(
                        model_path=resolved_model_path,
                        expected_num_gpus=expected_num_gpus,
                        result_path=result_path,
                        client_config=client_config,
                    )

                self._benchmark_results[server_idx] = server_results
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
        return self.log_files.copy()

    def upload_test_results_to_database(self) -> None:
        """Build OpenSearch documents and run the shared perf regression/upload pipeline."""
        if len(self._benchmark_results) != len(self.server_cases):
            raise RuntimeError(
                "VisualGen benchmark result count mismatch: "
                f"results={len(self._benchmark_results)}, servers={len(self.server_cases)}"
            )

        new_data_dict = {}
        result_idx = 0
        for server_idx, server_case in enumerate(self.server_cases):
            server_results = self._benchmark_results.get(server_idx)
            client_configs = server_case["client_configs"]
            if server_results is None or len(server_results) != len(client_configs):
                raise RuntimeError(
                    "VisualGen benchmark result count mismatch for "
                    f"{server_case['name']}: results={0 if server_results is None else len(server_results)}, "
                    f"clients={len(client_configs)}"
                )

            for client_idx, client_config in enumerate(client_configs):
                result_data = server_results.get(client_idx)
                if result_data is None:
                    raise RuntimeError(
                        f"Missing benchmark result for {server_case['name']} client index {client_idx}"
                    )

                new_data_dict[result_idx] = build_visual_gen_db_entry(
                    gpu_type=self.gpu_type,
                    model_name=str(server_case["model_name"]),
                    server_name=str(server_case["name"]),
                    server_config=server_case["server_config"],
                    client_config=client_config,
                    result_data=result_data,
                    extra_visual_gen_options_path=str(server_case["extra_visual_gen_options_path"]),
                )
                result_idx += 1

        extra_fields = {
            "s_stage_name": os.environ.get("stageName", ""),
            "s_test_list": self._test_param_labels,
        }
        process_and_upload_test_results(
            new_data_dict=new_data_dict,
            match_keys=get_visual_gen_match_keys(),
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
