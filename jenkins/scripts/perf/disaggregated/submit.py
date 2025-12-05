#!/usr/bin/env python3
import argparse
import os

import yaml


def get_hardware_config(config):
    hardware = config.get("hardware", {})

    num_ctx_servers = hardware.get("num_ctx_servers")
    num_gen_servers = hardware.get("num_gen_servers")
    gpus_per_node = hardware.get("gpus_per_node")
    gpus_per_ctx_server = hardware.get("gpus_per_ctx_server")
    gpus_per_gen_server = hardware.get("gpus_per_gen_server")

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
    job_workspace = env.get("job_workspace", "")
    worker_env_var = env.get("worker_env_var", "")
    server_env_var = env.get("server_env_var", "")
    benchmark_env_var = env.get("benchmark_env_var", "")

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
    }


def get_slurm_config(config, job_workspace):
    slurm = config.get("slurm", {})

    partition = slurm.get("partition", "")
    account = slurm.get("account", "")
    job_name = slurm.get("job_name", "disagg-test")
    output_path = slurm.get("output_path", os.path.join(job_workspace, "job-output.log"))

    return {
        "partition": partition,
        "account": account,
        "job_name": job_name,
        "output_path": output_path,
    }


def get_pytest_cmds(llmsrc, llm_models_root, job_workspace, stage_name, test_list_path):
    prefix = (
        f"LLM_ROOT={llmsrc} "
        f"LLM_BACKEND_ROOT={llmsrc}/triton_backend "
        f'__LUNOWUD="-thread_pool_size=12" '
        f"LLM_MODELS_ROOT={llm_models_root}"
    )
    llmapilaunch = f"{llmsrc}/tensorrt_llm/llmapi/trtllm-llmapi-launch"
    pytest = (
        f"pytest -v --perf --perf-log-formats=csv --perf-log-formats yaml "
        f"--timeout-method=thread --timeout=7200 "
        f"--rootdir {llmsrc}/tests/integration/defs "
        f"--test-prefix={stage_name} "
        f"--output-dir={job_workspace} "
        f"--csv={job_workspace}/report.csv "
        f"--junit-xml {job_workspace}/results.xml "
        f"-o junit_logging=out-err "
        f"--test-list={test_list_path} "
        f"--splitting-algorithm least_duration --splits 1 --group 1"
    )
    pytest_command = f"{prefix} {llmapilaunch} {pytest}"
    pytest_command_no_llmapi_launch = f"{prefix} {pytest}"
    return pytest_command, pytest_command_no_llmapi_launch


def remove_whitespace_lines(lines):
    return [line.strip() for line in lines if line.strip()]


def get_pytest_command_no_llmapilaunch(script_prefix_lines):
    pytest_command_line = None
    for line in script_prefix_lines:
        if "export pytestCommand=" in line:
            pytest_command_line = line
            break

    if not pytest_command_line:
        return ""

    # Replace pytestCommand with pytestCommandNoLLMAPILaunch
    replaced_line = pytest_command_line.replace("pytestCommand", "pytestCommandNoLLMAPILaunch")

    # Split by space, find and remove the substring with trtllm-llmapi-launch
    replaced_line_parts = replaced_line.split()
    replaced_line_parts_no_llmapi = [
        part for part in replaced_line_parts if "trtllm-llmapi-launch" not in part
    ]
    return " ".join(replaced_line_parts_no_llmapi)


def get_config_yaml(test_list_path, llm_src):
    with open(test_list_path, "r") as f:
        first_line = f.readline().strip()

    # Extract content between [ and ]
    if "[" not in first_line or "]" not in first_line:
        raise ValueError(
            f"Invalid test list format. Expected test name with brackets: {first_line}"
        )

    bracket_content = first_line.split("[")[-1].split("]")[0]

    # Split by - and get the second part (config_base_name)
    parts = bracket_content.split("-")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid test name format. Expected format: prefix-config_name, got: {bracket_content}"
        )

    config_base_name = parts[1]

    # Construct config yaml path
    config_yaml_path = os.path.join(
        llm_src, "tests", "scripts", "perf-sanity", f"{config_base_name}.yaml"
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

    if args.run_ci:
        config_yaml = get_config_yaml(args.test_list, args.llm_src)
    else:
        config_yaml = args.config_yaml

    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)

    hardware_config = get_hardware_config(config)
    print(f"Hardware configuration: {hardware_config}")

    env_config = get_env_config(config)
    print(f"Environment configuration: {env_config}")

    script_prefix_lines = []
    srun_args_lines = []
    if args.run_ci:
        with open(args.script_prefix, "r") as f:
            script_prefix_content = f.read()
        script_prefix_lines = script_prefix_content.split("\n")
        with open(args.srun_args, "r") as f:
            srun_args_content = f.read()
        srun_args_lines = srun_args_content.split()
    else:
        slurm_config = get_slurm_config(config, env_config["job_workspace"])
        print(f"SLURM configuration: {slurm_config}")

        # Get pytest commands for local mode
        test_list_path = os.path.join(env_config["job_workspace"], "test_list.txt")
        # Create job_workspace directory
        os.makedirs(env_config["job_workspace"], exist_ok=True)
        # Write test case to test_list.txt
        with open(test_list_path, "w") as f:
            config_basename = os.path.basename(config_yaml)
            yaml_name = config_basename.replace(".yaml", "").replace(".yml", "")
            f.write(f"perf/test_perf.py::test_perf[perf_sanity_upload-{yaml_name}]\n")

        # Get pytest command
        pytest_command, pytest_command_no_llmapi = get_pytest_cmds(
            env_config["llmsrc"],
            env_config["llm_models_root"],
            env_config["job_workspace"],
            args.stage_name,
            test_list_path,
        )

        trtllm_config_folder = os.path.dirname(config_yaml)

        # Build script-prefix
        script_prefix_lines.extend(
            [
                "#!/bin/bash",
                f"#SBATCH --nodes={hardware_config['total_nodes']}",
                f"#SBATCH --ntasks={hardware_config['total_gpus']}",
                f"#SBATCH --ntasks-per-node={hardware_config['gpus_per_node']}",
                "#SBATCH --time=04:00:00",
                f"#SBATCH --partition={slurm_config['partition']}",
                f"#SBATCH --account={slurm_config['account']}",
                f"#SBATCH --job-name={slurm_config['job_name']}",
                f"#SBATCH --output={slurm_config['output_path']}",
                "",
                'echo "Starting job $SLURM_JOB_ID on $SLURM_NODELIST"',
                "",
                "set -Eeuo pipefail",
                f"export jobWorkspace='{env_config['job_workspace']}'",
                f"export llmSrcNode='{env_config['llmsrc']}'",
                f"export stageName='{args.stage_name}'",
                "export perfMode='true'",
                f"export pytestCommand='{pytest_command}'",
                "export NVIDIA_IMEX_CHANNELS=0",
                (
                    "export NVIDIA_VISIBLE_DEVICES=$(seq -s, 0 "
                    "$(($(nvidia-smi --query-gpu=count -i 0 --format=noheader)-1)))"
                ),
                f"export TRTLLM_CONFIG_FOLDER='{trtllm_config_folder}'",
            ]
        )

        # Build srun-args
        srun_args_lines.extend(
            [
                f"--container-image={env_config['container']}",
                f"--container-workdir={env_config['workdir']}",
                f"--container-mounts={env_config['mounts']}",
                "--container-env=NVIDIA_IMEX_CHANNELS",
                "--mpi=pmix",
            ]
        )

    # Extract pytestCommand and generate pytestCommandNoLLMAPILaunch
    pytest_command_no_llmapi_launch = get_pytest_command_no_llmapilaunch(script_prefix_lines)

    script_prefix_lines.extend(
        [
            pytest_command_no_llmapi_launch,
            f'export pytestCommandWorker="unset UCX_TLS && {env_config["worker_env_var"]} $pytestCommand"',
            f'export pytestCommandDisaggServer="{env_config["server_env_var"]} $pytestCommandNoLLMAPILaunch"',
            f'export pytestCommandBenchmark="{env_config["benchmark_env_var"]} $pytestCommandNoLLMAPILaunch"',
            f"export runScript={args.run_sh}",
            f"export numCtxServers={hardware_config['num_ctx_servers']}",
            f"export numGenServers={hardware_config['num_gen_servers']}",
            f"export gpusPerNode={hardware_config['gpus_per_node']}",
            f"export gpusPerCtxServer={hardware_config['gpus_per_ctx_server']}",
            f"export gpusPerGenServer={hardware_config['gpus_per_gen_server']}",
            f"export nodesPerCtxServer={hardware_config['nodes_per_ctx_server']}",
            f"export nodesPerGenServer={hardware_config['nodes_per_gen_server']}",
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
