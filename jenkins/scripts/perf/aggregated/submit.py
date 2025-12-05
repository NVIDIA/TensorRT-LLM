#!/usr/bin/env python3
import argparse
import os

import yaml


def get_hardware_config(config):
    hardware = config.get("hardware", {})

    gpus_per_node = hardware.get("gpus_per_node")
    gpus_per_server = hardware.get("gpus_per_server")

    if None in [gpus_per_node, gpus_per_server]:
        raise ValueError(
            "Missing required hardware configuration: gpus_per_node or gpus_per_server"
        )

    total_nodes = gpus_per_server // gpus_per_node
    total_gpus = gpus_per_server

    return {
        "gpus_per_node": gpus_per_node,
        "gpus_per_server": gpus_per_server,
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
    # worker_env_var = env.get("worker_env_var", "")
    # server_env_var = env.get("server_env_var", "")

    return {
        "container": container,
        "mounts": mounts,
        "workdir": workdir,
        "llm_models_root": llm_models_root,
        "llmsrc": llmsrc,
        "build_wheel": build_wheel,
        "job_workspace": job_workspace,
        # "worker_env_var": worker_env_var,
        # "server_env_var": server_env_var,
    }


def get_slurm_config(config, job_workspace):
    slurm = config.get("slurm", {})

    partition = slurm.get("partition", "")
    account = slurm.get("account", "")
    job_name = slurm.get("job_name", "aggr-test")
    output_path = slurm.get("output_path", os.path.join(job_workspace, "job-output.log"))

    return {
        "partition": partition,
        "account": account,
        "job_name": job_name,
        "output_path": output_path,
    }


def get_pytest_cmd(llmsrc, llm_models_root, job_workspace, stage_name, test_list_path):
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
    return pytest_command


def remove_whitespace_lines(lines):
    return [line.strip() for line in lines if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM launch script for aggregated mode (local mode only)"
    )
    parser.add_argument("--config-yaml", required=True, help="Path to config YAML file")
    parser.add_argument("--draft-launch-sh", required=True, help="Path to draft-launch.sh script")
    parser.add_argument("--launch-sh", required=True, help="Path to output launch.sh script")
    parser.add_argument("--run-sh", required=True, help="Path to run.sh script")
    parser.add_argument("--stage-name", required=True, help="Stage name")
    parser.add_argument(
        "--build-wheel",
        action="store_true",
        default=False,
        help="Build TensorRT-LLM wheel before running tests",
    )

    args = parser.parse_args()

    with open(args.config_yaml, "r") as f:
        config = yaml.safe_load(f)

    hardware_config = get_hardware_config(config)
    print(f"Hardware configuration: {hardware_config}")

    env_config = get_env_config(config)
    print(f"Environment configuration: {env_config}")

    slurm_config = get_slurm_config(config, env_config["job_workspace"])
    print(f"SLURM configuration: {slurm_config}")

    # Get pytest commands for local mode
    test_list_path = os.path.join(env_config["job_workspace"], "test_list.txt")
    # Create job_workspace directory
    os.makedirs(env_config["job_workspace"], exist_ok=True)
    # Write test case to test_list.txt
    with open(test_list_path, "w") as f:
        config_basename = os.path.basename(args.config_yaml)
        yaml_name = config_basename.replace(".yaml", "").replace(".yml", "")
        f.write(f"perf/test_perf.py::test_perf[perf_sanity_upload-{yaml_name}]\n")

    # Get pytest command
    pytest_command = get_pytest_cmd(
        env_config["llmsrc"],
        env_config["llm_models_root"],
        env_config["job_workspace"],
        args.stage_name,
        test_list_path,
    )

    trtllm_config_folder = os.path.dirname(args.config_yaml)

    # Build script-prefix
    script_prefix_lines = [
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
        "export OPEN_SEARCH_DB_BASE_URL='http://gpuwa.nvidia.com'",
        f"export TRTLLM_CONFIG_FOLDER='{trtllm_config_folder}'",
        f"export runScript={args.run_sh}",
        f"export gpusPerNode={hardware_config['gpus_per_node']}",
        f"export gpusPerServer={hardware_config['gpus_per_server']}",
        f"export totalNodes={hardware_config['total_nodes']}",
        f"export totalGpus={hardware_config['total_gpus']}",
        f"export buildWheel={'true' if args.build_wheel else 'false'}",
    ]

    remove_whitespace_lines(script_prefix_lines)
    script_prefix = "\n".join(script_prefix_lines)

    # Build srun-args
    srun_args_lines = [
        f"--container-image={env_config['container']}",
        f"--container-workdir={env_config['workdir']}",
        f"--container-mounts={env_config['mounts']}",
        "--container-env=NVIDIA_IMEX_CHANNELS",
        "--container-env=OPEN_SEARCH_DB_BASE_URL",
        "--mpi=pmi2",
    ]

    remove_whitespace_lines(srun_args_lines)
    srun_args_lines = ["srunArgs=("] + [f"  '{line}'" for line in srun_args_lines] + [")"]
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
