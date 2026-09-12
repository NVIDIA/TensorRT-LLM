#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import glob
import math
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Submit aggregated benchmark job")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--config", type=str, help="Path to the configuration YAML file")
    group.add_argument(
        "-d", "--dir", type=str, help="Directory containing YAML configuration files"
    )
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory")
    parser.add_argument("--dry-run", action="store_true", help="Dry run, test purpose only")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def submit_job(config, config_path, log_dir, dry_run):
    slurm_config = config["slurm"]
    slurm_config.setdefault("extra_args", "")
    slurm_config.setdefault("set_segment", True)

    hw_config = config["hardware"]
    env_config = config["environment"]
    server_config = config["server"]
    benchmark_config = config["benchmark"]

    if "work_dir" in env_config and os.path.isdir(env_config["work_dir"]):
        script_dir = env_config["work_dir"]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))

    env_config.setdefault("trtllm_repo", "")
    env_config.setdefault("build_wheel", False)
    env_config.setdefault("cuda_architectures", "")
    env_config.setdefault("trtllm_wheel_path", "")
    env_config.setdefault("worker_env_var", "")

    profiling_config = config.get("profiling", {})
    profiling_config.setdefault("nsys_on", False)
    profiling_config.setdefault("profile_range", "10-30")

    gpus_per_node = hw_config["gpus_per_node"]
    tp_size = server_config["tp_size"]
    ep_size = server_config.get("ep_size", 1)
    pp_size = server_config.get("pp_size", 1)

    world_size = tp_size * pp_size
    total_nodes = math.ceil(world_size / gpus_per_node)

    isl = benchmark_config["input_length"]
    osl = benchmark_config["output_length"]
    max_batch_size = server_config["max_batch_size"]

    if log_dir is None:
        log_base = os.path.join(script_dir, "logs")
        date_prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_base = os.path.join(log_base, f"{date_prefix}/{isl}-{osl}")
        dir_suffix = f"agg_tp{tp_size}_ep{ep_size}_pp{pp_size}_batch{max_batch_size}"
        config_stem = os.path.splitext(os.path.basename(config_path))[0]
        if config_stem != "config":
            dir_suffix = f"{dir_suffix}_{config_stem}"
        log_dir = os.path.join(log_base, dir_suffix)

    if os.path.exists(log_dir):
        print(f"[WARNING] Removing existing log directory: {log_dir}")
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log will be saved to: {log_dir}")

    shutil.copy2(config_path, os.path.join(log_dir, "config.yaml"))

    extra_llm_api_path = server_config.get("extra_llm_api_options", "")
    if extra_llm_api_path and not os.path.isabs(extra_llm_api_path):
        extra_llm_api_path = os.path.join(script_dir, extra_llm_api_path)
    if extra_llm_api_path and os.path.isfile(extra_llm_api_path):
        shutil.copy2(
            extra_llm_api_path, os.path.join(log_dir, os.path.basename(extra_llm_api_path))
        )

    container_name = "agg-bench"
    container_mount_str = env_config["container_mount"]
    container_mount_str += f",{script_dir}:{script_dir}"

    worker_env = {}
    worker_env_var = env_config.get("worker_env_var", "")
    for var_string in worker_env_var.split():
        if "=" in var_string:
            key, val = var_string.split("=", 1)
            worker_env[key] = val

    if profiling_config["nsys_on"]:
        worker_env["TLLM_PROFILE_RECORD_GC"] = "1"
        worker_env["TLLM_NVTX_DEBUG"] = "1"
        worker_env["NSYS_MPI_STORE_TEAMS_PER_RANK"] = "1"
        worker_env["TLLM_PROFILE_START_STOP"] = profiling_config["profile_range"]

    # Build srun --export list. Values containing commas must be single-quoted
    # because srun uses comma as the delimiter between entries.
    export_parts = []
    for key, val in worker_env.items():
        if "," in val:
            export_parts.append(f"'{key}={val}'")
        else:
            export_parts.append(f"{key}={val}")
    export_str = ",".join(export_parts) if export_parts else "NONE"

    # Build server start command
    server_cmd_parts = [
        f"trtllm-serve {server_config['model_path']}",
        f"--max_batch_size {max_batch_size}",
        f"--max_num_tokens {server_config['max_num_tokens']}",
        f"--backend {server_config['backend']}",
    ]
    if extra_llm_api_path:
        server_cmd_parts.append(f"--extra_llm_api_options {extra_llm_api_path}")
    if ep_size > 1:
        server_cmd_parts.append(f"--ep_size {ep_size}")
    if server_config.get("trust_remote_code", False):
        server_cmd_parts.append("--trust_remote_code")
    server_cmd_parts.extend(
        [
            f"--gpus_per_node {gpus_per_node}",
            "--host 0.0.0.0",
            f"--port {server_config['port']}",
            f"--tp_size={tp_size}",
            f"--pp_size={pp_size}",
        ]
    )
    server_cmd = " ".join(server_cmd_parts)

    # Write start_server_cmds.sh
    start_server_cmd = " ".join(
        [
            "srun -l",
            f'--export="{export_str}"',
            f"--container-image {env_config['container_image']}",
            f"--container-name {container_name}",
            f"--container-mounts {container_mount_str}",
            "--no-container-mount-home --mpi=pmix --overlap",
            f"-N {total_nodes} -n {world_size}",
            f"--ntasks-per-node {gpus_per_node}",
            f"bash {os.path.join(script_dir, 'start_server.sh')}",
            f'"{server_cmd}"',
            log_dir,
            str(slurm_config.get("numa_bind", True)).lower(),
            str(profiling_config["nsys_on"]).lower(),
            f"&> {log_dir}/3_output_server.log &",
        ]
    )

    with open(os.path.join(log_dir, "start_server_cmds.sh"), "w") as f:
        f.write(start_server_cmd + "\n")

    # Write wait_server_cmds.sh
    wait_server_cmd = " ".join(
        [
            "srun -l",
            f"--container-name={container_name}",
            f"--container-mounts={container_mount_str}",
            "--mpi=pmix --overlap -N 1 -n 1",
            f"bash {os.path.join(script_dir, 'wait_server.sh')}",
            "<HOSTNAME>",
            str(server_config["port"]),
            f"&> {log_dir}/4_wait_server.log",
        ]
    )

    with open(os.path.join(log_dir, "wait_server_cmds.sh"), "w") as f:
        f.write(wait_server_cmd + "\n")

    # Write client_cmds.sh
    num_prompts = benchmark_config.get("num_prompts", 1024)
    max_concurrency = benchmark_config.get("max_concurrency", 256)
    streaming_flag = "" if benchmark_config.get("streaming", True) else "--non-streaming"
    dataset_name = benchmark_config.get("dataset_name", "random")

    bench_cmd = " ".join(
        [
            "srun -l",
            f"--container-name={container_name}",
            f"--container-mounts={container_mount_str}",
            "--mpi=pmix --overlap -N 1 -n 1",
            "python -m tensorrt_llm.serve.scripts.benchmark_serving",
            f"--model {server_config['model_path']}",
            "--backend openai",
            "--host <HOSTNAME>",
            f"--port {server_config['port']}",
            f"--dataset-name {dataset_name}",
            f"--random-input-len {isl}",
            f"--random-output-len {osl}",
            "--random-ids",
            f"--num-prompts {num_prompts}",
            f"--max-concurrency {max_concurrency}",
            "--trust-remote-code",
            "--ignore-eos",
            "--save-result",
            f"--result-filename {log_dir}/benchmark_serving_results.json",
            "--percentile-metrics ttft,tpot,itl",
            streaming_flag,
            f"&> {log_dir}/5_bench.log",
        ]
    )

    done_cmd = f"echo ${{SLURM_JOB_NODELIST}} > {log_dir}/6_done_${{SLURM_JOB_ID}}.txt"

    with open(os.path.join(log_dir, "client_cmds.sh"), "w") as f:
        f.write(bench_cmd + "\n")
        f.write(done_cmd + "\n")

    # Resolve slurm script_file path
    slurm_script_file = slurm_config["script_file"]
    if not os.path.isabs(slurm_script_file):
        slurm_script_file = os.path.join(script_dir, slurm_script_file)

    if not os.path.exists(slurm_script_file):
        print(f"[ERROR] SLURM script file not found: {slurm_script_file}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        "sbatch",
        f"--partition={slurm_config['partition']}",
        f"--account={slurm_config['account']}",
        f"--time={slurm_config['job_time']}",
        f"--job-name={slurm_config['job_name']}",
        f"--nodes={total_nodes}",
        f"--ntasks={world_size}",
        f"--ntasks-per-node={gpus_per_node}",
        *([] if not slurm_config["set_segment"] else [f"--segment={total_nodes}"]),
        f"--output={log_dir}/slurm-%j.out",
        f"--error={log_dir}/slurm-%j.err",
        *([arg for arg in slurm_config["extra_args"].split() if arg]),
        slurm_script_file,
        "--work-dir",
        script_dir,
        "--full-logdir",
        log_dir,
        "--container-name",
        container_name,
        "--container-mount",
        container_mount_str,
        "--container-image",
        env_config["container_image"],
        "--trtllm-repo",
        env_config["trtllm_repo"],
        "--build-wheel",
        str(env_config["build_wheel"]).lower(),
        "--cuda-architectures",
        env_config.get("cuda_architectures", ""),
        "--trtllm-wheel-path",
        env_config.get("trtllm_wheel_path", ""),
    ]

    if dry_run:
        print("[WARNING] Dry run mode, will not submit the job.")
        print("sbatch command:")
        print(" ".join(cmd))
        return
    else:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job: {e}", file=sys.stderr)
            sys.exit(1)


def main():
    args = parse_args()

    if args.config:
        config_files = [args.config]
    else:
        yaml_pattern = os.path.join(args.dir, "*.yaml")
        config_files = sorted(glob.glob(yaml_pattern))
        if not config_files:
            print(f"No YAML files found in directory: {args.dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(config_files)} YAML file(s) in {args.dir}")

    for config_file in config_files:
        print(f"Processing: {config_file}")
        try:
            config = load_config(config_file)
            submit_job(config, config_file, args.log_dir, args.dry_run)
            print(f"Successfully submitted job for: {config_file}\n")
        except Exception as e:
            traceback.print_exc()
            print(f"Error processing {config_file}: {e}", file=sys.stderr)
            continue


if __name__ == "__main__":
    main()
