#!/usr/bin/env python3
import argparse
import os

import yaml


def get_hardware_config(config, benchmark_mode):
    hardware = config.get("hardware", {})
    worker_config = config.get("worker_config", {})

    num_ctx_servers = 0 if "gen_only" in benchmark_mode else hardware.get("num_ctx_servers")
    num_gen_servers = hardware.get("num_gen_servers")
    gpus_per_node = hardware.get("gpus_per_node")

    # Get gpus_per_ctx_server and gpus_per_gen_server from worker_config's tensor_parallel_size
    ctx_config = worker_config.get("ctx", {})
    gen_config = worker_config.get("gen", {})
    ctx_tp = ctx_config.get("tensor_parallel_size", 1)
    ctx_pp = ctx_config.get("pipeline_parallel_size", 1)
    ctx_cp = ctx_config.get("context_parallel_size", 1)
    gpus_per_ctx_server = ctx_tp * ctx_pp * ctx_cp
    gen_tp = gen_config.get("tensor_parallel_size", 1)
    gen_pp = gen_config.get("pipeline_parallel_size", 1)
    gen_cp = gen_config.get("context_parallel_size", 1)
    gpus_per_gen_server = gen_tp * gen_pp * gen_cp

    if None in [
        num_ctx_servers,
        num_gen_servers,
        gpus_per_node,
        gpus_per_ctx_server,
        gpus_per_gen_server,
    ]:
        raise ValueError("Missing required hardware configuration")

    # Calculate nodes per server
    nodes_per_ctx_server = (gpus_per_ctx_server + gpus_per_node - 1) // gpus_per_node
    nodes_per_gen_server = (gpus_per_gen_server + gpus_per_node - 1) // gpus_per_node

    gpus_per_node_per_ctx_server = min(gpus_per_ctx_server, gpus_per_node)
    gpus_per_node_per_gen_server = min(gpus_per_gen_server, gpus_per_node)

    total_nodes = num_ctx_servers * nodes_per_ctx_server + num_gen_servers * nodes_per_gen_server
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


def get_env_config(config):
    env = config.get("environment", {})

    container = env.get("container_image", "")
    mounts = env.get("container_mount", "")
    workdir = env.get("container_workdir", "")
    llm_models_root = env.get("llm_models_root", "")
    llmsrc = env.get("trtllm_repo", "")
    build_wheel = env.get("build_wheel", False)
    # Use work_dir as job_workspace
    job_workspace = env.get("work_dir", "")
    worker_env_var = env.get("worker_env_var", "")
    server_env_var = env.get("server_env_var", "")
    benchmark_env_var = env.get("benchmark_env_var", "")
    open_search_db_base_url = env.get("open_search_db_base_url", "")

    return {
        "container": container,
        "mounts": mounts,
        "workdir": workdir,
        "llm_models_root": llm_models_root,
        "llmsrc": llmsrc,
        "build_wheel": build_wheel,
        "job_workspace": job_workspace,
        "worker_env_var": worker_env_var,
        "server_env_var": server_env_var,
        "benchmark_env_var": benchmark_env_var,
        "open_search_db_base_url": open_search_db_base_url,
    }


def get_benchmark_config(config):
    benchmark = config.get("benchmark", {})

    mode = benchmark.get("mode", "e2e")
    concurrency_str = benchmark.get("concurrency_list", "1")
    concurrency = int(concurrency_str) if isinstance(concurrency_str, str) else concurrency_str

    return {
        "mode": mode,
        "concurrency": concurrency,
    }


def remove_whitespace_lines(lines):
    return [line.strip() for line in lines if line.strip()]


def get_pytest_commands(script_prefix_lines):
    # Get worker, disagg_server, benchmark pytest commands from pytest command.
    # Worker pytest command is pytest command with trtllm-llmapi-launch and
    # without --csv, --cov, --periodic flags.
    # Disagg_server pytest command is pytest command without trtllm-llmapi-launch
    # and without --csv, --cov, --periodic flags.
    # Benchmark pytest command is pytest command without trtllm-llmapi-launch
    # and with --csv, --cov, --periodic flags.
    pytest_command_line = None
    for line in script_prefix_lines:
        if "export pytestCommand=" in line:
            pytest_command_line = line
            break

    if not pytest_command_line:
        return "", "", ""

    def split_pytest_command_line(command_line):
        # After pytest, there are six types of substrings:
        # Type 1: --xxx=yyy  (long option with value, self-contained)
        # Type 2: --xxx=     (long option with empty value, self-contained)
        # Type 3: --xxx      (long option flag, no value)
        # Type 4: --xxx yyy  (long option with value as next arg)
        # Type 5: -x yyy     (short single-letter option with value as next arg)
        # Type 6: -x         (short option flag, e.g., -v, -vv)
        parts = command_line.split()
        pytest_index = None
        for idx, part in enumerate(parts):
            if "pytest" == part:
                pytest_index = idx
                break
        if pytest_index is None:
            return parts

        grouped_parts = parts[: pytest_index + 1]
        i = pytest_index + 1
        while i < len(parts):
            part = parts[i]
            has_next = i + 1 < len(parts)
            next_is_value = has_next and not parts[i + 1].startswith("-")

            # Type 1 & 2: --xxx=yyy or --xxx= (self-contained, has '=')
            if part.startswith("--") and "=" in part:
                grouped_parts.append(part)
                i += 1
                continue

            # Type 4: --xxx yyy (long option with value as next arg)
            if part.startswith("--") and next_is_value:
                grouped_parts.append(f"{part} {parts[i + 1]}")
                i += 2
                continue

            # Type 3: --xxx (long option flag)
            if part.startswith("--"):
                grouped_parts.append(part)
                i += 1
                continue

            # Type 5: -x yyy (short single-letter option with value as next arg)
            # Only single letter after dash, e.g., -o, not -vv
            if part.startswith("-") and len(part) == 2 and next_is_value:
                grouped_parts.append(f"{part} {parts[i + 1]}")
                i += 2
                continue

            # Type 6: -x (short option flag, including combined like -vv)
            if part.startswith("-"):
                grouped_parts.append(part)
                i += 1
                continue

            # Other parts (shouldn't happen after pytest, but handle gracefully)
            grouped_parts.append(part)
            i += 1

        return grouped_parts

    def is_llmapi_launch(part):
        return "trtllm-llmapi-launch" in part

    def is_output_file_part(part):
        return any(flag in part for flag in ("--csv", "--cov", "--periodic"))

    worker_line = pytest_command_line.replace("pytestCommand", "partialPytestCommandWorker")
    worker_parts = [
        part for part in split_pytest_command_line(worker_line) if not is_output_file_part(part)
    ]
    worker_pytest_command = " ".join(worker_parts)

    disagg_server_line = pytest_command_line.replace(
        "pytestCommand", "partialPytestCommandDisaggServer"
    )
    disagg_server_parts = [
        part
        for part in split_pytest_command_line(disagg_server_line)
        if not is_llmapi_launch(part) and not is_output_file_part(part)
    ]
    disagg_server_pytest_command = " ".join(disagg_server_parts)

    benchmark_line = pytest_command_line.replace("pytestCommand", "partialPytestCommandBenchmark")
    benchmark_parts = [
        part for part in split_pytest_command_line(benchmark_line) if not is_llmapi_launch(part)
    ]
    benchmark_pytest_command = " ".join(benchmark_parts)

    return (
        worker_pytest_command,
        disagg_server_pytest_command,
        benchmark_pytest_command,
    )


def get_config_yaml(test_list_path, llm_src):
    with open(test_list_path, "r") as f:
        first_line = f.readline().strip()

    if "[" not in first_line or "]" not in first_line:
        raise ValueError(
            f"Invalid test list format. Expected test name with brackets: {first_line}"
        )
    bracket_content = first_line.split("[")[-1].split("]")[0]
    parts = bracket_content.split("-")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid test name format. Expected format: prefix-config_name, got: {bracket_content}"
        )

    # parts[0] is the prefix, parts[1:] is the config name
    if "disagg" not in parts[0]:
        raise ValueError(
            f"Invalid test name format. Expected format: disagg-config_name, got: {bracket_content}"
        )
    config_base_name = "-".join(parts[1:])
    config_yaml_path = os.path.join(
        llm_src,
        "tests",
        "integration",
        "defs",
        "perf",
        "disagg",
        "test_configs",
        "disagg",
        "perf-sanity",
        f"{config_base_name}.yaml",
    )
    if not os.path.exists(config_yaml_path):
        raise FileNotFoundError(f"Config file not found: {config_yaml_path}")
    return config_yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM launch script for both CI and local modes"
    )
    parser.add_argument(
        "--run-ci",
        action="store_true",
        default=False,
        help="Run in CI mode (true) or local mode (false)",
    )
    parser.add_argument("--draft-launch-sh", required=True, help="Path to draft-launch.sh script")
    parser.add_argument("--launch-sh", required=True, help="Path to output launch.sh script")
    parser.add_argument("--run-sh", required=True, help="Path to slurm_run.sh script")
    parser.add_argument("--install-sh", required=True, help="Path to slurm_install.sh script")

    # Optional arguments for local mode
    parser.add_argument("--config-yaml", default="", help="Path to config YAML file")
    parser.add_argument("--stage-name", default="", help="Stage name (optional, local mode only)")

    # Optional arguments for CI mode
    parser.add_argument("--llm-src", default="", help="Path to LLM source code")
    parser.add_argument("--test-list", default="", help="Path to test list file")
    parser.add_argument(
        "--script-prefix",
        default="",
        help="Launch script prefix file path (optional, CI mode only)",
    )
    parser.add_argument(
        "--srun-args",
        default="",
        help="Path to file containing srun args (optional, CI mode only)",
    )

    args = parser.parse_args()

    config_yaml = get_config_yaml(args.test_list, args.llm_src)

    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)

    # Determine install script path
    install_script = args.install_sh

    env_config = get_env_config(config)
    print(f"Environment configuration: {env_config}")

    benchmark_config = get_benchmark_config(config)
    print(f"Benchmark configuration: {benchmark_config}")
    benchmark_mode = benchmark_config["mode"]

    hardware_config = get_hardware_config(config, benchmark_mode)
    print(f"Hardware configuration: {hardware_config}")

    script_prefix_lines = []
    srun_args_lines = []

    with open(args.script_prefix, "r") as f:
        script_prefix_content = f.read()
    script_prefix_lines = script_prefix_content.split("\n")
    with open(args.srun_args, "r") as f:
        srun_args_content = f.read()

    srun_args_lines = srun_args_content.split()

    # Extract pytestCommand and generate partial pytest commands
    (
        worker_pytest_command,
        disagg_server_pytest_command,
        benchmark_pytest_command,
    ) = get_pytest_commands(script_prefix_lines)

    # Build worker env vars, add extra env vars for gen_only mode
    worker_env_vars = env_config["worker_env_var"]
    server_env_vars = env_config["server_env_var"]
    if "gen_only" in benchmark_config["mode"]:
        concurrency = benchmark_config["concurrency"]
        worker_env_vars = (
            "TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1 "
            f"TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1 "
            f"TLLM_BENCHMARK_REQ_QUEUES_SIZE={concurrency} {worker_env_vars}"
        )
        server_env_vars = f"TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1 {server_env_vars}"
        script_prefix_lines.append("export TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1")
        srun_args_lines.append("--container-env=TRTLLM_DISAGG_BENCHMARK_GEN_ONLY")

    script_prefix_lines.extend(
        [
            worker_pytest_command,
            disagg_server_pytest_command,
            benchmark_pytest_command,
            f'export pytestCommandWorker="unset UCX_TLS && {worker_env_vars} $partialPytestCommandWorker"',
            f'export pytestCommandDisaggServer="{server_env_vars} $partialPytestCommandDisaggServer"',
            f'export pytestCommandBenchmark="{env_config["benchmark_env_var"]} $partialPytestCommandBenchmark"',
            f"export runScript={args.run_sh}",
            f"export installScript={install_script}",
            f"export configYamlPath={config_yaml}",
            f"export numCtxServers={hardware_config['num_ctx_servers']}",
            f"export numGenServers={hardware_config['num_gen_servers']}",
            f"export gpusPerNode={hardware_config['gpus_per_node']}",
            f"export gpusPerCtxServer={hardware_config['gpus_per_ctx_server']}",
            f"export gpusPerGenServer={hardware_config['gpus_per_gen_server']}",
            f"export nodesPerCtxServer={hardware_config['nodes_per_ctx_server']}",
            f"export nodesPerGenServer={hardware_config['nodes_per_gen_server']}",
            f"export gpusPerfNodePerfCtxServer={hardware_config['gpus_per_node_per_ctx_server']}",
            f"export gpusPerfNodePerfGenServer={hardware_config['gpus_per_node_per_gen_server']}",
            f"export totalNodes={hardware_config['total_nodes']}",
            f"export totalGpus={hardware_config['total_gpus']}",
        ]
    )

    remove_whitespace_lines(script_prefix_lines)
    script_prefix = "\n".join(script_prefix_lines)

    remove_whitespace_lines(srun_args_lines)
    srun_args_lines.extend(
        [
            "--container-env=DISAGG_SERVING_TYPE",
            "--container-env=pytestCommand",
        ]
    )
    srun_args_lines = ["srunArgs=("] + [f'  "{line}"' for line in srun_args_lines] + [")"]
    srun_args = "\n".join(srun_args_lines)

    with open(args.draft_launch_sh, "r") as f:
        draft_launch_content = f.read()
    draft_launch_lines = draft_launch_content.split("\n")
    remove_whitespace_lines(draft_launch_lines)
    draft_launch_content = "\n".join(draft_launch_lines)

    with open(args.launch_sh, "w") as f:
        f.write(f"{script_prefix}\n{srun_args}\n{draft_launch_content}")

    print(f"Launch script generated at: {args.launch_sh}")
    print(f"Launch script:\n{script_prefix}\n{srun_args}\n{draft_launch_content}")


if __name__ == "__main__":
    main()
