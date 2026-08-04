#!/usr/bin/env python3
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
import math
import os
import re

import yaml

AGG_CONFIG_FOLDER = "tests/scripts/perf-sanity/aggregated"
DISAGG_CONFIG_FOLDER = "tests/scripts/perf-sanity/disaggregated"


# --------------------------------------------------------------------------- #
# Test list parsing
# --------------------------------------------------------------------------- #
def parse_test_case_name(test_list_path, llm_src, split_group=0):
    """Parse the selected line of the test list.

    Returns (config_yaml_path, server_name, benchmark_mode, runtime_mode).
    See the module docstring for the supported test name shapes.
    """
    with open(test_list_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Test list is empty: {test_list_path}")

    if split_group > 0:
        if split_group > len(lines):
            raise ValueError(
                f"split_group {split_group} exceeds number of tests in test list ({len(lines)})"
            )
        line = lines[split_group - 1]
    else:
        line = lines[0]

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


def get_env_config(config, runtime_mode, benchmark_mode, server_name):
    """Get worker / server / benchmark env vars from the yaml.

    Aggregated yaml stores env vars per server config under
    `server_configs[i].server_env_var`. Disaggregated yaml stores them at the
    top-level `environment.{worker,server,benchmark}_env_var`.

    ctx_only is a hybrid: the launch path is aggregated, but the yaml is the
    disagg one, so the agg launch's "server_env_var" comes from
    `environment.worker_env_var`.

    Returns: {worker_env_var, server_env_var, benchmark_env_var}.
    """
    env = config.get("environment", {}) or {}
    if runtime_mode == "aggregated":
        if benchmark_mode == "ctx_only":
            return {
                "worker_env_var": env.get("worker_env_var", "") or "",
                "server_env_var": env.get("worker_env_var", "") or "",
                "benchmark_env_var": env.get("benchmark_env_var", "") or "",
            }
        agg_server_env_var = ""
        for sc in config.get("server_configs", []) or []:
            if sc.get("name") == server_name:
                agg_server_env_var = sc.get("server_env_var", "") or ""
                break
        return {
            "worker_env_var": "",
            "server_env_var": agg_server_env_var,
            "benchmark_env_var": "",
        }
    return {
        "worker_env_var": env.get("worker_env_var", "") or "",
        "server_env_var": env.get("server_env_var", "") or "",
        "benchmark_env_var": env.get("benchmark_env_var", "") or "",
    }


def get_benchmark_config(config):
    benchmark = config.get("benchmark", {}) or {}
    concurrency_str = benchmark.get("concurrency_list", "1")
    concurrency = int(concurrency_str) if isinstance(concurrency_str, str) else concurrency_str
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
        help="1-indexed split group id. Selects the N-th test from the test list.",
    )
    parser.add_argument("--stage-name", default="", help="Stage name (for logging / GPU detect)")

    args = parser.parse_args()

    config_yaml, server_name, benchmark_mode, runtime_mode = parse_test_case_name(
        args.test_list, args.llm_src, args.split_group
    )

    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)

    # Recover test_case_name (the bracketed pytest test id) for the per-test
    # output dir — same line/split logic as parse_test_case_name.
    with open(args.test_list, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    sel = lines[args.split_group - 1] if args.split_group > 0 else lines[0]
    test_case_name = sel.split("[")[-1].split("]")[0] if "[" in sel else ""

    hardware_config = get_hardware_config(config, runtime_mode, benchmark_mode, server_name)
    env_config = get_env_config(config, runtime_mode, benchmark_mode, server_name)
    benchmark_config = get_benchmark_config(config)

    print(f"runtime_mode: {runtime_mode!r}")
    print(f"benchmark_mode: {benchmark_mode!r}")
    print(f"server_name: {server_name!r}")
    print(f"Hardware configuration: {hardware_config}")
    print(f"Environment configuration: {env_config}")
    print(f"Benchmark configuration: {benchmark_config}")

    with open(args.script_prefix, "r") as f:
        script_prefix_content = f.read()
    script_prefix_lines = script_prefix_content.split("\n")

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

    is_gb300 = "GB300" in args.stage_name.upper()
    is_b200 = "B200" in args.stage_name.upper() and "GB200" not in args.stage_name.upper()

    if runtime_mode == "aggregated":
        # Aggregated (incl. ctx_only): single pytestCommand built from the
        # matched server_config's server_env_var (regular agg) or the disagg
        # yaml's environment.worker_env_var (ctx_only). The prefix runs on
        # every rank before trtllm-llmapi-launch dispatches to pytest (rank 0)
        # or mgmn_worker_node (others).
        server_env_var = env_config["server_env_var"]
        if server_env_var.strip():
            pytest_command_with_env = (
                f'export pytestCommand="{server_env_var} $partialPytestCommand"'
            )
        else:
            pytest_command_with_env = 'export pytestCommand="$partialPytestCommand"'

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
        base_worker_env_vars = (
            f"FLASHINFER_JIT_DIR=/tmp/flashinfer_jit_cache_\\${{SLURM_LOCALID}} "
            f"HF_HOME=/tmp/hf_home "
            f"{env_config['worker_env_var']}"
        )
        ctx_worker_env_vars = base_worker_env_vars
        gen_worker_env_vars = base_worker_env_vars
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
            ctx_worker_env_vars = (
                f"TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1 {ctx_worker_env_vars}"
            )
            gen_worker_env_vars = (
                f"TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1 "
                f"TLLM_BENCHMARK_REQ_QUEUES_SIZE={concurrency} {gen_worker_env_vars}"
            )

        if is_gb300:
            ucx_tls_cmd = "export UCX_TLS=cuda_copy,cuda_ipc,sm,self,tcp &&"
        elif is_b200:
            ucx_tls_cmd = "export UCX_TLS=^ib &&"
        else:
            ucx_tls_cmd = "unset UCX_TLS UCX_NET_DEVICES &&"
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
        srun_args_lines.extend(
            [
                "--container-env=DISAGG_SERVING_TYPE",
                "--container-env=pytestCommand",
            ]
        )

    script_prefix_lines = remove_whitespace_lines(script_prefix_lines)
    script_prefix = "\n".join(script_prefix_lines)

    srun_args_lines = remove_whitespace_lines(srun_args_lines)
    srun_args_lines = ["srunArgs=("] + [f'  "{line}"' for line in srun_args_lines] + [")"]
    srun_args = "\n".join(srun_args_lines)

    with open(args.draft_launch_sh, "r") as f:
        draft_launch_content = f.read()
    draft_launch_lines = remove_whitespace_lines(draft_launch_content.split("\n"))
    draft_launch_content = "\n".join(draft_launch_lines)

    with open(args.launch_sh, "w") as f:
        f.write(f"{script_prefix}\n{srun_args}\n{draft_launch_content}")

    print(f"Launch script generated at: {args.launch_sh}")
    print(f"Launch script:\n{script_prefix}\n{srun_args}\n{draft_launch_content}")


if __name__ == "__main__":
    main()
