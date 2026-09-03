#!/usr/bin/env python3
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

"""Generate the SLURM launch script for multi-node PerfSanity tests (CI mode).

Unified replacement for jenkins/scripts/perf/aggregated/submit.py and
jenkins/scripts/perf/disaggregated/submit.py.

Three test shapes are supported (all flow through the same parsing logic):
  1. Multi-node aggregated:        aggr[_upload]-{config_base}-{server_name}
        runtime_mode = "aggregated", benchmark_mode = None
  2. Multi-node ctx_only disagg:   aggr[_upload]-ctx_only-{config_base}
        runtime_mode = "aggregated", benchmark_mode = "ctx_only"
        (reads disagg yaml, but launches via the aggregated single-pytest path
         using the ctx worker's parallel sizes)
  3. Multi-node disagg e2e/gen:    disagg[_upload]-{e2e|gen_only}-{config_base}
        runtime_mode = "disaggregated", benchmark_mode in {"e2e", "gen_only"}

Test name → yaml folder mapping mirrors test_perf_sanity.py:parse_test_string.
"""

import argparse
import heapq
import json
import math
import os
import re
import shlex
import sys

import yaml
from benchmark_utils import parse_positive_concurrency
from cluster_env import get_ucx_env_cmd, gpu_type_from_stage_name


def _import_precheck_config(llm_src):
    """Import the pure-stdlib precheck config module from the repo tree.

    It is the single owner of the gate's enable policy and timeout formulas.
    """
    path = os.path.join(llm_src, "tests", "scripts", "perf-sanity", "cache_transceiver_precheck")
    if path not in sys.path:
        sys.path.insert(0, path)
    import precheck_config

    return precheck_config


AGG_CONFIG_FOLDER = "tests/scripts/perf-sanity/aggregated"
DISAGG_CONFIG_FOLDER = "tests/scripts/perf-sanity/disaggregated"


# --------------------------------------------------------------------------- #
# Test list parsing
# --------------------------------------------------------------------------- #
def _read_test_list_lines(test_list_path):
    """Read runnable entries from a generated test list.

    Args:
        test_list_path: Path to the generated test-list file.

    Returns:
        Non-empty, non-comment test-list lines in source order.

    Raises:
        ValueError: If the test list has no runnable entries.
    """
    with open(test_list_path, "r") as f:
        lines = []
        for line in f:
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith("#"):
                lines.append(stripped_line)
    if not lines:
        raise ValueError(f"Test list is empty: {test_list_path}")
    return lines


def _pytest_command_tokens(script_prefix_lines):
    """Parse the exported pytest command from a launch-script prefix.

    Args:
        script_prefix_lines: Lines from the launch-script prefix.

    Returns:
        Shell-parsed pytest command tokens, or an empty list when the export is absent.
    """
    pytest_command_line = next(
        (line for line in script_prefix_lines if "export pytestCommand=" in line), ""
    )
    if not pytest_command_line:
        return []
    command = pytest_command_line.split("=", 1)[1].strip()
    if len(command) >= 2 and command[0] == command[-1] and command[0] in ('"', "'"):
        command = command[1:-1]
    return shlex.split(command)


def _pytest_option(tokens, option):
    """Return a pytest option's value from command tokens.

    Args:
        tokens: Shell-parsed pytest command tokens.
        option: Option name to find, including its leading dashes.

    Returns:
        The option value, or ``None`` when the option or its value is absent.
    """
    for index, token in enumerate(tokens):
        if token == option:
            return tokens[index + 1] if index + 1 < len(tokens) else None
        if token.startswith(f"{option}="):
            return token.split("=", 1)[1]
    return None


def _test_nodeid(test_line):
    """Strip test-list markers from a line to recover pytest's node ID.

    Args:
        test_line: Test-list entry, optionally followed by execution markers.

    Returns:
        The pytest node ID from the entry.
    """
    return re.split(
        r"\s+(?:XFAIL|SKIP|UNSTABLE|TIMEOUT)(?=[\s(]|$)",
        test_line,
        maxsplit=1,
    )[0]


def _test_marker(test_line):
    """Return a test-list execution marker, or ``None`` when absent."""
    line = test_line.partition("#")[0].strip()
    match = re.search(r"\s+(XFAIL|SKIP|UNSTABLE|TIMEOUT)(?=[\s(]|$)", line)
    return match.group(1) if match else None


def selected_test_is_skip_waived(selected_test_line, waives_file, test_prefix=None):
    """Whether pytest will skip the selected case before executing its body.

    The CI pipeline merges remote waives into the repository waives file
    before invoking this launcher. Mirror the exact-nodeid SKIP decision here
    so a skipped test does not run an otherwise unrelated precheck first.
    """
    if _test_marker(selected_test_line) == "SKIP":
        return True

    selected_nodeid = _test_nodeid(selected_test_line).strip()
    try:
        waive_lines = _read_test_list_lines(waives_file)
    except (FileNotFoundError, ValueError):
        return False

    for line in waive_lines:
        if _test_marker(line) != "SKIP":
            continue
        waived_nodeid = _test_nodeid(line).strip()
        if waived_nodeid.startswith("full:"):
            scope, separator, waived_nodeid = waived_nodeid[5:].partition("/")
            if not separator or not test_prefix:
                continue
            # Match the platform-prefix handling in test_list_parser. SM
            # waives require runtime GPU discovery and remain pytest-owned.
            platform_prefix = test_prefix.split("-", 1)[0]
            if scope.startswith("sm") or platform_prefix not in scope:
                continue
        if waived_nodeid == selected_nodeid:
            return True
    return False


def _load_pytest_split_durations(tokens, llm_src):
    """Load pytest-split duration data using the launcher's path fallback.

    Args:
        tokens: Shell-parsed pytest command tokens.
        llm_src: TensorRT-LLM source-tree path used for the repository fallback.

    Returns:
        A pair containing the duration mapping and the path it was loaded from.

    Raises:
        FileNotFoundError: If neither the configured path nor its repository fallback exists.
        ValueError: If the duration data is not a mapping or legacy list of pairs.
    """
    durations_option = _pytest_option(tokens, "--durations-path")
    if durations_option:
        durations_path = durations_option
        if not os.path.exists(durations_path):
            durations_path = os.path.join(
                llm_src,
                "tests",
                "integration",
                "defs",
                os.path.basename(durations_option),
            )
    else:
        durations_path = os.path.join(llm_src, "tests", "integration", "defs", ".test_durations")

    try:
        with open(durations_path, "r") as durations_file:
            durations = json.load(durations_file)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"pytest-split durations file not found: {durations_path}"
        ) from error
    if isinstance(durations, list):
        durations = dict(durations)
    if not isinstance(durations, dict):
        raise ValueError(f"Invalid pytest-split durations file: {durations_path}")
    return durations, durations_path


def _select_least_duration_group(lines, durations, splits, group):
    """Mirror pytest-split's ``LeastDurationAlgorithm`` exactly.

    Args:
        lines: Test-list entries to distribute among the split groups.
        durations: Mapping from pytest node IDs to recorded durations.
        splits: Number of duration-balanced groups to create.
        group: One-indexed group to return.

    Returns:
        Entries assigned to the requested group, in their original test-list order.

    Raises:
        ValueError: If ``splits`` is less than one or ``group`` is outside its range.
    """
    if splits < 1:
        raise ValueError(f"pytest --splits must be >= 1, got {splits}")
    if group < 1 or group > splits:
        raise ValueError(f"pytest --group must be in [1, {splits}], got {group}")

    nodeids = [_test_nodeid(line) for line in lines]
    relevant_durations = {
        nodeid: float(durations[nodeid]) for nodeid in nodeids if nodeid in durations
    }
    average_duration = (
        sum(relevant_durations.values()) / len(relevant_durations) if relevant_durations else 1.0
    )
    items = [
        (line, nodeid, relevant_durations.get(nodeid, average_duration), original_index)
        for original_index, (line, nodeid) in enumerate(zip(lines, nodeids))
    ]

    # pytest-split first sorts by item name, then performs a stable descending
    # duration sort. It greedily places each item in the least-loaded group.
    items.sort(key=lambda item: item[1])
    items.sort(key=lambda item: item[2], reverse=True)
    selected = [[] for _ in range(splits)]
    group_heap = [(0.0, group_index) for group_index in range(splits)]
    heapq.heapify(group_heap)
    for line, _nodeid, duration, original_index in items:
        group_duration, group_index = heapq.heappop(group_heap)
        selected[group_index].append((original_index, line))
        heapq.heappush(group_heap, (group_duration + duration, group_index))

    return [line for _original_index, line in sorted(selected[group - 1], key=lambda item: item[0])]


def select_test_case_line(test_list_path, llm_src, script_prefix_lines, split_group=0):
    """Select the same test as the pytest-split shard in ``pytestCommand``."""
    lines = _read_test_list_lines(test_list_path)
    if split_group <= 0:
        return lines[0]

    tokens = _pytest_command_tokens(script_prefix_lines)
    splits_option = _pytest_option(tokens, "--splits")
    group_option = _pytest_option(tokens, "--group")
    algorithm = _pytest_option(tokens, "--splitting-algorithm")
    if splits_option is None or group_option is None:
        if split_group > len(lines):
            raise ValueError(
                f"split_group {split_group} exceeds number of tests in test list ({len(lines)})"
            )
        return lines[split_group - 1]
    if algorithm != "least_duration":
        raise ValueError(
            "Multi-node perf launcher only supports pytest-split's least_duration "
            f"algorithm, got {algorithm!r}"
        )

    splits = int(splits_option)
    pytest_group = int(group_option)
    if pytest_group != split_group:
        raise ValueError(
            f"submit.py split_group={split_group} disagrees with pytest --group={pytest_group}"
        )
    durations, durations_path = _load_pytest_split_durations(tokens, llm_src)
    selected = _select_least_duration_group(lines, durations, splits, pytest_group)
    if len(selected) != 1:
        raise ValueError(
            "Multi-node perf launch requires exactly one test in each pytest-split "
            f"group, but group {pytest_group}/{splits} selected {len(selected)} tests "
            f"using {durations_path}: {selected}"
        )
    print(
        f"Selected pytest-split group {pytest_group}/{splits} test using "
        f"{durations_path}: {_test_nodeid(selected[0])}"
    )
    return selected[0]


def parse_test_case_name(llm_src, selected_line):
    """Parse the selected test-list line.

    Returns (config_yaml_path, server_name, benchmark_mode, runtime_mode).
    See the module docstring for the supported test name shapes.
    """
    line = selected_line

    if "[" not in line or "]" not in line:
        raise ValueError(f"Invalid test list format. Expected name with brackets: {line}")
    bracket_content = line.split("[")[-1].split("]")[0]
    parts = bracket_content.split("-")

    if len(parts) < 2:
        raise ValueError(f"Invalid test name (need at least prefix and config): {bracket_content}")

    prefix = parts[0]
    if "disagg" in prefix:
        if len(parts) < 3:
            raise ValueError(
                f"Invalid disagg test format. Expected disagg[_upload]-{{e2e|gen_only}}-"
                f"{{config_base}}, got: {bracket_content}"
            )
        benchmark_mode = parts[1]
        if benchmark_mode not in ("e2e", "gen_only"):
            raise ValueError(
                f"Invalid disagg benchmark_mode: {benchmark_mode}. Expected 'e2e' or 'gen_only'."
            )
        runtime_mode = "disaggregated"
        server_name = None
        config_base_name = "-".join(parts[2:])
        config_yaml_path = os.path.join(llm_src, DISAGG_CONFIG_FOLDER, f"{config_base_name}.yaml")
    elif "aggr" in prefix:
        if len(parts) > 2 and parts[1] == "ctx_only":
            # ctx_only: aggr[_upload]-ctx_only-{config_base}; reads disagg yaml.
            benchmark_mode = "ctx_only"
            runtime_mode = "aggregated"
            server_name = None
            config_base_name = "-".join(parts[2:])
            config_yaml_path = os.path.join(
                llm_src, DISAGG_CONFIG_FOLDER, f"{config_base_name}.yaml"
            )
        else:
            # Regular agg: aggr[_upload]-{config_base}-{server_name}.
            # config_base_name is a single label and server_name is everything
            # after it — mirrors test_perf_sanity.py:parse_test_string so the
            # launch script and the test runner agree on the config path.
            if len(parts) < 3:
                raise ValueError(
                    f"Invalid agg test format. Expected aggr[_upload]-{{config_base}}-"
                    f"{{server_name}}, got: {bracket_content}"
                )
            benchmark_mode = None
            runtime_mode = "aggregated"
            config_base_name = parts[1]
            server_name = "-".join(parts[2:])
            config_yaml_path = os.path.join(llm_src, AGG_CONFIG_FOLDER, f"{config_base_name}.yaml")
    else:
        raise ValueError(
            f"Invalid test name prefix '{prefix}'. Expected starts-with 'aggr' or 'disagg'."
        )

    if not os.path.exists(config_yaml_path):
        raise FileNotFoundError(f"Config file not found: {config_yaml_path}")

    return config_yaml_path, server_name, benchmark_mode, runtime_mode


# --------------------------------------------------------------------------- #
# Hardware / env / benchmark config (unified across agg and disagg)
# --------------------------------------------------------------------------- #
def get_hardware_config(config, runtime_mode, benchmark_mode, server_name):
    """Compute the hardware layout. Mirrors local/submit.py:get_hardware_config.

    Aggregated (incl. ctx_only) returns:
        gpus_per_node, gpus_per_server, nodes_per_server, gpus_per_node_per_server,
        total_nodes, total_gpus, world_size
    Disaggregated returns:
        num_ctx_servers, num_gen_servers, gpus_per_node,
        gpus_per_ctx_server, gpus_per_gen_server,
        nodes_per_ctx_server, nodes_per_gen_server,
        gpus_per_node_per_ctx_server, gpus_per_node_per_gen_server,
        total_nodes, total_gpus
    """
    hardware = config.get("hardware", {}) or {}
    gpus_per_node = hardware.get("gpus_per_node")
    if gpus_per_node is None:
        raise ValueError("hardware.gpus_per_node is required")

    if benchmark_mode == "ctx_only":
        # ctx_only reads disagg yaml; size the launch from worker_config.ctx.
        worker_config = config.get("worker_config", {}) or {}
        ctx_config = worker_config.get("ctx", {}) or {}
        if not ctx_config:
            raise ValueError("worker_config.ctx is required for ctx_only mode")
        tp = ctx_config.get("tensor_parallel_size", 1)
        pp = ctx_config.get("pipeline_parallel_size", 1)
        cp = ctx_config.get("context_parallel_size", 1)
        gpus_per_server = ctx_config.get("world_size") or (tp * pp * cp)
    elif runtime_mode == "aggregated":
        # Regular agg: match server_configs by name.
        server_configs = config.get("server_configs", []) or []
        server_config = next((sc for sc in server_configs if sc.get("name") == server_name), None)
        if server_config is None:
            raise ValueError(f"server_config not found for name: {server_name}")
        tp = server_config.get("tensor_parallel_size", 1)
        pp = server_config.get("pipeline_parallel_size", 1)
        cp = server_config.get("context_parallel_size", 1)
        gpus_per_server = server_config.get("world_size") or (tp * pp * cp)
    else:
        # Disaggregated: separate ctx + gen workers.
        worker_config = config.get("worker_config", {}) or {}
        ctx_config = worker_config.get("ctx", {}) or {}
        gen_config = worker_config.get("gen", {}) or {}

        # gen_only_no_context comes from the yaml's benchmark.mode, not the
        # test name (test name is always "gen_only" for both gen_only and
        # gen_only_no_context tests). When set, ctx workers are not launched.
        yaml_mode = (config.get("benchmark", {}) or {}).get("mode", "")
        is_gen_only_no_context = benchmark_mode == "gen_only" and "gen_only_no_context" in yaml_mode
        num_ctx_servers = 0 if is_gen_only_no_context else hardware.get("num_ctx_servers")
        num_gen_servers = hardware.get("num_gen_servers")

        ctx_tp = ctx_config.get("tensor_parallel_size", 1)
        ctx_pp = ctx_config.get("pipeline_parallel_size", 1)
        ctx_cp = ctx_config.get("context_parallel_size", 1)
        gpus_per_ctx_server = ctx_tp * ctx_pp * ctx_cp
        gen_tp = gen_config.get("tensor_parallel_size", 1)
        gen_pp = gen_config.get("pipeline_parallel_size", 1)
        gen_cp = gen_config.get("context_parallel_size", 1)
        gpus_per_gen_server = gen_tp * gen_pp * gen_cp

        if None in [num_ctx_servers, num_gen_servers, gpus_per_ctx_server, gpus_per_gen_server]:
            raise ValueError("Missing required disagg hardware configuration")

        nodes_per_ctx_server = math.ceil(gpus_per_ctx_server / gpus_per_node)
        nodes_per_gen_server = math.ceil(gpus_per_gen_server / gpus_per_node)
        gpus_per_node_per_ctx_server = min(gpus_per_ctx_server, gpus_per_node)
        gpus_per_node_per_gen_server = min(gpus_per_gen_server, gpus_per_node)

        total_nodes = (
            num_ctx_servers * nodes_per_ctx_server + num_gen_servers * nodes_per_gen_server
        )
        total_gpus = total_nodes * gpus_per_node

        return {
            "num_ctx_servers": num_ctx_servers,
            "num_gen_servers": num_gen_servers,
            "gpus_per_node": gpus_per_node,
            "gpus_per_ctx_server": gpus_per_ctx_server,
            "gpus_per_gen_server": gpus_per_gen_server,
            "nodes_per_ctx_server": nodes_per_ctx_server,
            "nodes_per_gen_server": nodes_per_gen_server,
            "gpus_per_node_per_ctx_server": gpus_per_node_per_ctx_server,
            "gpus_per_node_per_gen_server": gpus_per_node_per_gen_server,
            "total_nodes": total_nodes,
            "total_gpus": total_gpus,
        }

    # Aggregated (regular or ctx_only) shared layout.
    nodes_per_server = math.ceil(gpus_per_server / gpus_per_node)
    total_nodes = nodes_per_server
    gpus_per_node_per_server = min(gpus_per_server, gpus_per_node)
    world_size = total_nodes * gpus_per_node_per_server
    return {
        "gpus_per_node": gpus_per_node,
        "gpus_per_server": gpus_per_server,
        "nodes_per_server": nodes_per_server,
        "gpus_per_node_per_server": gpus_per_node_per_server,
        "total_nodes": total_nodes,
        "total_gpus": total_nodes * gpus_per_node,
        "world_size": world_size,
    }


def _join_env(*parts):
    """Space-join non-empty env-var strings (drops falsy entries)."""
    return " ".join(p for p in parts if p)


def get_env_config(config, runtime_mode, benchmark_mode, server_name):
    """Get worker / server / benchmark env vars from the yaml.

    Aggregated yaml stores env vars per server config under
    `server_configs[i].server_env_var`. Disaggregated yaml stores them at the
    top-level `environment.{worker,server,benchmark}_env_var`, plus optional
    `environment.{ctx,gen}_worker_env_var` for role-specific extras (appended
    to the shared `worker_env_var`).

    ctx_only is a hybrid: the launch path is aggregated, but the yaml is the
    disagg one, so the agg launch's "server_env_var" comes from
    `environment.worker_env_var` (merged with ctx-side extras when present).

    Returns: {worker_env_var (shared, back-compat),
              ctx_worker_env_var, gen_worker_env_var,
              server_env_var, benchmark_env_var}.
    """
    env = config.get("environment", {}) or {}
    common = env.get("worker_env_var", "") or ""
    ctx_extra = env.get("ctx_worker_env_var", "") or ""
    gen_extra = env.get("gen_worker_env_var", "") or ""
    ctx_env = _join_env(common, ctx_extra)
    gen_env = _join_env(common, gen_extra)
    if runtime_mode == "aggregated":
        if benchmark_mode == "ctx_only":
            return {
                "worker_env_var": common,
                "ctx_worker_env_var": ctx_env,
                "gen_worker_env_var": gen_env,
                # ctx_only launches through the aggregated single-pytest path;
                # the ctx-merged env is what actually runs.
                "server_env_var": ctx_env,
                "benchmark_env_var": env.get("benchmark_env_var", "") or "",
            }
        agg_server_env_var = ""
        for sc in config.get("server_configs", []) or []:
            if sc.get("name") == server_name:
                agg_server_env_var = sc.get("server_env_var", "") or ""
                break
        return {
            "worker_env_var": "",
            "ctx_worker_env_var": "",
            "gen_worker_env_var": "",
            "server_env_var": agg_server_env_var,
            "benchmark_env_var": "",
        }
    return {
        "worker_env_var": common,
        "ctx_worker_env_var": ctx_env,
        "gen_worker_env_var": gen_env,
        "server_env_var": env.get("server_env_var", "") or "",
        "benchmark_env_var": env.get("benchmark_env_var", "") or "",
    }


def get_benchmark_config(config):
    benchmark = config.get("benchmark", {}) or {}
    concurrency = parse_positive_concurrency(benchmark.get("concurrency_list", "1"))
    return {
        "mode": benchmark.get("mode", ""),
        "concurrency": concurrency,
    }


# --------------------------------------------------------------------------- #
# pytestCommand splitting
# --------------------------------------------------------------------------- #
def _split_pytest_command_line(command_line):
    """Group the pytest tail of `command_line` into self-contained tokens.

    After `pytest`, six argument shapes appear:
      Type 1: --xxx=yyy   (long option, value via '=')
      Type 2: --xxx=      (long option, empty value)
      Type 3: --xxx       (long option flag)
      Type 4: --xxx yyy   (long option, value as next token)
      Type 5: -x yyy      (short option, value as next token)
      Type 6: -x / -vv    (short flag(s))
    Tokens before `pytest` are kept as-is (the env-var prefix, `pytest` itself).
    """
    parts = command_line.split()
    pytest_index = None
    for idx, part in enumerate(parts):
        if part == "pytest":
            pytest_index = idx
            break
    if pytest_index is None:
        return parts

    grouped = parts[: pytest_index + 1]
    i = pytest_index + 1
    while i < len(parts):
        part = parts[i]
        has_next = i + 1 < len(parts)
        next_is_value = has_next and not parts[i + 1].startswith("-")

        if part.startswith("--") and "=" in part:  # Type 1 & 2
            grouped.append(part)
            i += 1
        elif part.startswith("--") and next_is_value:  # Type 4
            grouped.append(f"{part} {parts[i + 1]}")
            i += 2
        elif part.startswith("--"):  # Type 3
            grouped.append(part)
            i += 1
        elif part.startswith("-") and len(part) == 2 and next_is_value:  # Type 5
            grouped.append(f"{part} {parts[i + 1]}")
            i += 2
        elif part.startswith("-"):  # Type 6
            grouped.append(part)
            i += 1
        else:
            grouped.append(part)
            i += 1
    return grouped


def get_pytest_commands(script_prefix_lines, runtime_mode):
    """Emit the partial pytestCommand variants needed by the launch path.

    Finds the inbound `export pytestCommand=...` line and rewrites it.

    Aggregated (incl. ctx_only) returns a 4-tuple:
        (agg_partial_line, "", "", "")
    where agg_partial_line is the original line with `pytestCommand` renamed
    to `partialPytestCommand`.

    Disaggregated returns a 4-tuple:
        ("", worker_line, disagg_server_line, benchmark_line)
    where:
      - worker_line:        rename to partialPytestCommandWorker; drop --csv/--cov/--periodic
      - disagg_server_line: rename to partialPytestCommandDisaggServer; drop trtllm-llmapi-launch
                            and --csv/--cov/--periodic
      - benchmark_line:     rename to partialPytestCommandBenchmark; drop trtllm-llmapi-launch
    """
    pytest_command_line = next(
        (ln for ln in script_prefix_lines if "export pytestCommand=" in ln), None
    )
    if not pytest_command_line:
        return ("", "", "", "")

    if runtime_mode == "aggregated":
        agg_line = pytest_command_line.replace("pytestCommand", "partialPytestCommand")
        return (agg_line, "", "", "")

    def _is_llmapi_launch(part):
        return "trtllm-llmapi-launch" in part

    def _is_output_file_part(part):
        return any(flag in part for flag in ("--csv", "--cov", "--periodic"))

    worker_line = pytest_command_line.replace("pytestCommand", "partialPytestCommandWorker")
    worker_parts = [
        p for p in _split_pytest_command_line(worker_line) if not _is_output_file_part(p)
    ]
    worker_pytest_command = " ".join(worker_parts)

    disagg_server_line = pytest_command_line.replace(
        "pytestCommand", "partialPytestCommandDisaggServer"
    )
    disagg_server_parts = [
        p
        for p in _split_pytest_command_line(disagg_server_line)
        if not _is_llmapi_launch(p) and not _is_output_file_part(p)
    ]
    disagg_server_pytest_command = " ".join(disagg_server_parts)

    benchmark_line = pytest_command_line.replace("pytestCommand", "partialPytestCommandBenchmark")
    benchmark_parts = [
        p for p in _split_pytest_command_line(benchmark_line) if not _is_llmapi_launch(p)
    ]
    benchmark_pytest_command = " ".join(benchmark_parts)

    return ("", worker_pytest_command, disagg_server_pytest_command, benchmark_pytest_command)


def get_test_output_dir(script_prefix_lines, test_case_name):
    """Build the per-test output directory from the inbound pytestCommand.

    Picks `--output-dir` out of the inbound pytestCommand and appends the
    test_case_name — same shape as
    PerfSanityTestConfig.get_commands: <output_dir>/<test_case_name>.
    """
    pytest_command_line = next(
        (ln for ln in script_prefix_lines if "export pytestCommand=" in ln), ""
    )
    if not pytest_command_line:
        return ""
    m = re.search(r'--output-dir[=\s]+"?([^"\s]+)"?', pytest_command_line)
    if not m:
        return ""
    output_dir = m.group(1)
    return os.path.join(output_dir, test_case_name) if test_case_name else output_dir


class _PytestCommandEnvMissing(ValueError):
    """A valid pytestCommand does not provide the requested leading variable."""


def extract_pytest_command_env(script_prefix_lines, name):
    """Read a leading environment assignment from the exported pytest command."""
    line = next((ln for ln in script_prefix_lines if "export pytestCommand=" in ln), None)
    if line is None:
        raise ValueError("launch prefix does not export pytestCommand")
    try:
        outer_tokens = shlex.split(line)
    except ValueError as e:
        raise ValueError(f"cannot parse exported pytestCommand: {e}") from e
    command_assignment = next(
        (token for token in outer_tokens if token.startswith("pytestCommand=")), None
    )
    if command_assignment is None:
        raise ValueError("launch prefix has a malformed pytestCommand export")
    command = command_assignment.partition("=")[2]
    try:
        command_tokens = shlex.split(command)
    except ValueError as e:
        raise ValueError(f"cannot parse pytestCommand payload: {e}") from e
    for token in command_tokens:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token):
            break
        key, value = token.split("=", 1)
        if key == name:
            return value
    raise _PytestCommandEnvMissing(
        f"pytestCommand does not set leading environment variable {name}"
    )


def _resolve_llm_models_root(script_prefix_lines):
    """Resolve the precheck model root from pytestCommand or the submitter env."""
    try:
        return extract_pytest_command_env(script_prefix_lines, "LLM_MODELS_ROOT")
    except _PytestCommandEnvMissing as e:
        fallback = os.environ.get("LLM_MODELS_ROOT")
        if fallback:
            return fallback
        # Fail closed when the precheck is enabled: without the model root it
        # cannot reproduce serving's KV shape and model-specific defaults, so
        # disabling the gate here would silently run an unvalidated workload.
        raise ValueError(
            f"{e}; LLM_MODELS_ROOT is also absent from the submitter environment "
            "(pytestCommand is assembled by getPytestBaseCommandLine in L0_Test.groovy)"
        ) from e


def remove_whitespace_lines(lines):
    return [line.strip() for line in lines if line.strip()]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate the SLURM launch script for multi-node aggregated and "
            "disaggregated PerfSanity tests (CI mode)."
        )
    )
    parser.add_argument("--draft-launch-sh", required=True, help="Path to draft-launch.sh script")
    parser.add_argument("--launch-sh", required=True, help="Path to output launch.sh script")
    parser.add_argument("--run-sh", required=True, help="Path to slurm_run.sh script")
    parser.add_argument("--install-sh", required=True, help="Path to slurm_install.sh script")
    parser.add_argument("--llm-src", required=True, help="Path to LLM source code")
    parser.add_argument("--test-list", required=True, help="Path to test list file")
    parser.add_argument(
        "--script-prefix",
        required=True,
        help="Launch script prefix file path",
    )
    parser.add_argument(
        "--srun-args",
        required=True,
        help="Path to file containing srun args",
    )
    parser.add_argument(
        "--split-group",
        type=int,
        default=0,
        help=(
            "1-indexed pytest-split group id. Selects the same duration-balanced test as pytest."
        ),
    )
    parser.add_argument("--stage-name", default="", help="Stage name (for logging / GPU detect)")
    parser.add_argument(
        "--cluster-name",
        default="",
        help="Slurm cluster name as resolved by the Jenkins pipeline "
        "(bloom SlurmPartition.clusterName, e.g. gcp-nrt, aws-cmh). "
        "Used with the GPU type to pick UCX env settings.",
    )

    args = parser.parse_args()

    with open(args.script_prefix, "r") as f:
        script_prefix_content = f.read()
    script_prefix_lines = script_prefix_content.split("\n")

    selected_test_line = select_test_case_line(
        args.test_list,
        args.llm_src,
        script_prefix_lines,
        args.split_group,
    )
    pytest_tokens = _pytest_command_tokens(script_prefix_lines)
    selected_test_skipped = selected_test_is_skip_waived(
        selected_test_line,
        os.path.join(args.llm_src, "tests", "integration", "test_lists", "waives.txt"),
        test_prefix=_pytest_option(pytest_tokens, "--test-prefix"),
    )
    if selected_test_skipped:
        print("Selected test is SKIP-waived; cache-transceiver precheck will not run")
    config_yaml, server_name, benchmark_mode, runtime_mode = parse_test_case_name(
        args.llm_src,
        selected_test_line,
    )

    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)

    test_case_name = (
        selected_test_line.split("[")[-1].split("]")[0] if "[" in selected_test_line else ""
    )

    hardware_config = get_hardware_config(config, runtime_mode, benchmark_mode, server_name)
    env_config = get_env_config(config, runtime_mode, benchmark_mode, server_name)
    benchmark_config = get_benchmark_config(config)

    print(f"runtime_mode: {runtime_mode!r}")
    print(f"benchmark_mode: {benchmark_mode!r}")
    print(f"server_name: {server_name!r}")
    print(f"Hardware configuration: {hardware_config}")
    print(f"Environment configuration: {env_config}")
    print(f"Benchmark configuration: {benchmark_config}")

    with open(args.srun_args, "r") as f:
        srun_args_content = f.read()
    srun_args_lines = srun_args_content.split()

    (
        agg_pytest_command,
        worker_pytest_command,
        disagg_server_pytest_command,
        benchmark_pytest_command,
    ) = get_pytest_commands(script_prefix_lines, runtime_mode)
    test_output_dir = get_test_output_dir(script_prefix_lines, test_case_name)

    gpu_type = gpu_type_from_stage_name(args.stage_name)
    ucx_tls_cmd = get_ucx_env_cmd(
        runtime_mode,
        hardware_config,
        args.cluster_name,
        gpu_type,
    )
    if ucx_tls_cmd:
        print(f"UCX env: cluster={args.cluster_name!r} gpu={gpu_type!r} -> {ucx_tls_cmd!r}")

    if runtime_mode == "aggregated":
        # Aggregated (incl. ctx_only): single pytestCommand built from the
        # matched server_config's server_env_var (regular agg) or the disagg
        # yaml's environment.worker_env_var (ctx_only). The prefix runs on
        # every rank before trtllm-llmapi-launch dispatches to pytest (rank 0)
        # or mgmn_worker_node (others).
        server_env_var = env_config["server_env_var"]
        pytest_command = _join_env(ucx_tls_cmd, server_env_var, "$partialPytestCommand")
        pytest_command_with_env = f'export pytestCommand="{pytest_command}"'

        script_prefix_lines.extend(
            [
                agg_pytest_command,
                pytest_command_with_env,
                f"export runScript={args.run_sh}",
                f"export installScript={args.install_sh}",
                f"export configYamlPath={config_yaml}",
                f"export gpusPerNode={hardware_config['gpus_per_node']}",
                f"export gpusPerNodePerServer={hardware_config['gpus_per_node_per_server']}",
                f"export nodesPerServer={hardware_config['nodes_per_server']}",
                f"export totalNodes={hardware_config['total_nodes']}",
                f"export world_size={hardware_config['world_size']}",
                f"export testOutputDir={test_output_dir}",
            ]
        )
        srun_args_lines.append("--container-env=pytestCommand")
    else:
        # Disaggregated (e2e or gen_only).
        base_prefix = (
            "FLASHINFER_JIT_DIR=/tmp/flashinfer_jit_cache_\\${SLURM_LOCALID} HF_HOME=/tmp/hf_home"
        )
        # ctx / gen env vars: shared worker_env_var + optional per-role extras
        # from the yaml. get_env_config already merged them.
        ctx_worker_env_vars = f"{base_prefix} {env_config['ctx_worker_env_var']}".rstrip()
        gen_worker_env_vars = f"{base_prefix} {env_config['gen_worker_env_var']}".rstrip()
        server_env_vars = env_config["server_env_var"]

        # gen_only_no_context comes from yaml's benchmark.mode, not the test
        # name — see get_hardware_config.
        yaml_mode = benchmark_config.get("mode", "")
        if benchmark_mode == "gen_only" and "gen_only_no_context" in yaml_mode:
            gen_worker_env_vars = f"TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1 {gen_worker_env_vars}"
            server_env_vars = f"TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1 {server_env_vars}"
            script_prefix_lines.append("export TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1")
            srun_args_lines.append("--container-env=TRTLLM_DISAGG_BENCHMARK_GEN_ONLY")
        elif benchmark_mode == "gen_only":
            concurrency = benchmark_config.get("concurrency", 1)
            # GEN worker only: the same flag on the CTX worker has been seen to
            # hang gen_only runs with KV blocks never released.
            gen_worker_env_vars = (
                f"TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1 "
                f"TLLM_BENCHMARK_REQ_QUEUES_SIZE={concurrency} {gen_worker_env_vars}"
            )

        ucx_tls_server_cmd = ucx_tls_cmd

        pytest_common_vars = ""
        script_prefix_lines.extend(
            [
                worker_pytest_command,
                disagg_server_pytest_command,
                benchmark_pytest_command,
                f'export PYTEST_COMMON_VARS="{pytest_common_vars}"',
                f'export CTX_WORKER_ENV_VARS="{ctx_worker_env_vars}"',
                f'export GEN_WORKER_ENV_VARS="{gen_worker_env_vars}"',
                f'export SERVER_ENV_VARS="{server_env_vars}"',
                f'export BENCHMARK_ENV_VARS="{env_config["benchmark_env_var"]}"',
                f'export pytestCommandCTXWorker="{ucx_tls_cmd} $CTX_WORKER_ENV_VARS'
                ' $PYTEST_COMMON_VARS $partialPytestCommandWorker"',
                f'export pytestCommandGENWorker="{ucx_tls_cmd} $GEN_WORKER_ENV_VARS'
                ' $PYTEST_COMMON_VARS $partialPytestCommandWorker"',
                f'export pytestCommandDisaggServer="{ucx_tls_server_cmd}'
                ' $SERVER_ENV_VARS $PYTEST_COMMON_VARS $partialPytestCommandDisaggServer"',
                f'export pytestCommandBenchmark="{ucx_tls_cmd} $BENCHMARK_ENV_VARS'
                ' $PYTEST_COMMON_VARS $partialPytestCommandBenchmark"',
                f"export runScript={args.run_sh}",
                f"export installScript={args.install_sh}",
                f"export configYamlPath={config_yaml}",
                f"export numCtxServers={hardware_config['num_ctx_servers']}",
                f"export numGenServers={hardware_config['num_gen_servers']}",
                f"export gpusPerNode={hardware_config['gpus_per_node']}",
                f"export gpusPerCtxServer={hardware_config['gpus_per_ctx_server']}",
                f"export gpusPerGenServer={hardware_config['gpus_per_gen_server']}",
                f"export nodesPerCtxServer={hardware_config['nodes_per_ctx_server']}",
                f"export nodesPerGenServer={hardware_config['nodes_per_gen_server']}",
                f"export gpusPerNodePerCtxServer={hardware_config['gpus_per_node_per_ctx_server']}",
                f"export gpusPerNodePerGenServer={hardware_config['gpus_per_node_per_gen_server']}",
                f"export totalNodes={hardware_config['total_nodes']}",
                f"export totalGpus={hardware_config['total_gpus']}",
                f"export testOutputDir={test_output_dir}",
            ]
        )

        # Cache-transceiver network precheck: runs BEFORE the real ctx/gen
        # servers with the same instance topology, and reuses the exact
        # $ucx_tls_cmd / $CTX_WORKER_ENV_VARS / $GEN_WORKER_ENV_VARS strings
        # of the worker steps so the UCX environment matches by construction.
        # Enable/kill-switch policy and timeouts live in precheck_config
        # (single owner, shared with the local flow).
        pcfg = _import_precheck_config(args.llm_src)
        precheck_enabled = pcfg.precheck_enabled(config)
        precheck_will_run = precheck_enabled and not selected_test_skipped
        # The model root is only consumed by the precheck (auto KV-cache-manager
        # resolution needs the model config). Fail fast only when the precheck
        # will actually run; otherwise degrade to a warning so stages whose
        # pytestCommand does not carry LLM_MODELS_ROOT inline keep submitting.
        llm_models_root = None
        if precheck_enabled:
            try:
                llm_models_root = _resolve_llm_models_root(script_prefix_lines)
            except ValueError as e:
                if precheck_will_run:
                    raise
                print(
                    f"WARNING: {e}; "
                    "cache-transceiver precheck is skipped for this config so continuing"
                )
        script_prefix_lines.extend(
            pcfg.precheck_prefix_lines(
                config,
                benchmark_mode,
                config_path_expr=f"$llmSrcNode/{os.path.relpath(config_yaml, args.llm_src)}",
                ucx_tls_cmd=ucx_tls_cmd,
                max_world=max(
                    hardware_config["gpus_per_ctx_server"],
                    hardware_config["gpus_per_gen_server"],
                ),
                stage_name=args.stage_name,
                llm_models_root=llm_models_root,
                skip_precheck=selected_test_skipped,
            )
        )
        srun_args_lines.extend(
            ["--container-env=DISAGG_SERVING_TYPE", "--container-env=pytestCommand"]
        )
        if precheck_enabled:
            srun_args_lines.append("--container-env=LLM_MODELS_ROOT")

    script_prefix_lines = remove_whitespace_lines(script_prefix_lines)
    script_prefix = "\n".join(script_prefix_lines)

    srun_args_lines = remove_whitespace_lines(srun_args_lines)
    srun_args_lines = ["srunArgs=("] + [f'  "{line}"' for line in srun_args_lines] + [")"]
    srun_args = "\n".join(srun_args_lines)

    with open(args.draft_launch_sh, "r") as f:
        draft_launch_content = f.read()
    draft_launch_lines = remove_whitespace_lines(draft_launch_content.split("\n"))
    draft_launch_content = "\n".join(draft_launch_lines)

    # The disagg draft calls run_cache_transceiver_precheck; splice in the gate
    # function library ahead of it (single owner: precheck_config).
    gate_content = ""
    if runtime_mode == "disaggregated":
        gate_content = pcfg.gate_library_content(args.draft_launch_sh, args.llm_src)

    with open(args.launch_sh, "w") as f:
        f.write(f"{script_prefix}\n{srun_args}\n{gate_content}{draft_launch_content}")

    print(f"Launch script generated at: {args.launch_sh}")
    print(f"Launch script:\n{script_prefix}\n{srun_args}\n{draft_launch_content}")


if __name__ == "__main__":
    main()
