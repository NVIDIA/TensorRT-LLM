#!/usr/bin/env python3

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='Submit disaggregated benchmark job')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-c',
                       '--config',
                       type=str,
                       help='Path to the configuration YAML file')
    group.add_argument('-d',
                       '--dir',
                       type=str,
                       help='Directory containing YAML configuration files')
    parser.add_argument('--log-dir',
                        type=str,
                        default=None,
                        help='Log directory')
    parser.add_argument('--dry-run',
                        action='store_true',
                        help='Dry run the Python part, test purpose only')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_worker_config(worker_config, output_path):
    """Save worker config to a separate YAML file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(worker_config, f, default_flow_style=False)


def calculate_nodes(world_size, num_servers, gpus_per_node):
    """Calculate required nodes based on world size and server count."""
    return math.ceil(world_size * num_servers / gpus_per_node)


def allocate_gpus(
    total_nodes: int,
    gpus_per_node: int,
    num_gen_servers: int,
    num_ctx_servers: int,
    gen_world_size: int,
    ctx_world_size: int,
    base_port: int = 8000,
) -> List[Dict[str, Any]]:
    allocations = {}
    hostnames = [f"<node{i}_placeholder>" for i in range(total_nodes)]

    global_gpu_cursor = 0

    def get_gpu_location(gpus_per_node: int):
        node_id = global_gpu_cursor // gpus_per_node
        local_gpu_id = global_gpu_cursor % gpus_per_node
        return node_id, local_gpu_id

    def assign_server(server_allocation: Dict[str, Any], world_size: int,
                      gpus_per_node: int):
        nonlocal global_gpu_cursor
        for _ in range(world_size):
            node_id, gpu_id = get_gpu_location(gpus_per_node)
            hostname = hostnames[node_id]
            if hostname not in server_allocation["nodes"]:
                server_allocation["nodes"][hostname] = []
            server_allocation["nodes"][hostname].append(gpu_id)
            global_gpu_cursor += 1

    def assign_servers(
        server_allocations: Dict[str, Any],
        server_type: str,
        num_servers: int,
        world_size: int,
        gpus_per_node: int,
    ):
        if server_type not in server_allocations:
            server_allocations[server_type] = {}
        for i in range(num_servers):
            server_allocation = {
                "port": base_port + i,
                "nodes": {},
            }
            assign_server(server_allocation, world_size, gpus_per_node)
            server_allocations[server_type][i] = server_allocation

    assign_servers(allocations, "GEN", num_gen_servers, gen_world_size,
                   gpus_per_node)
    assign_servers(allocations, "CTX", num_ctx_servers, ctx_world_size,
                   gpus_per_node)

    return allocations


def convert_allocations_to_server_config(allocations, server_port=8333):
    generation_servers = {}
    context_servers = {}
    server_hostname = None

    for server_type in allocations.keys():
        num_servers = len(allocations[server_type])
        urls = []
        for server_id in allocations[server_type].keys():
            instance = allocations[server_type][server_id]
            urls.append(
                f"{list(instance['nodes'].keys())[0]}:{instance['port']}")

        server_config_entry = {'num_instances': num_servers, 'urls': urls}

        if server_type == "GEN":
            generation_servers = server_config_entry
            server_hostname = urls[0].split(':')[0]
            if allocations[server_type][server_id]['port'] == server_port:
                server_port += 1  # Avoid port conflict
        elif server_type == "CTX":
            context_servers = server_config_entry

    server_config = {
        'backend': 'pytorch',
        'hostname': server_hostname,
        'port': server_port,
        'context_servers': context_servers,
        'generation_servers': generation_servers
    }
    return server_config


def convert_envs_to_str(env_vars: Dict[str, str]) -> str:
    return ','.join([f"{key}='{value}'" for key, value in env_vars.items()])


def replace_env_in_file(log_dir, file_path, env_var):
    with open(file_path, 'r', encoding='utf-8') as f:
        config_content = f.read()

    for env_name, env_value in env_var.items():
        file_content = config_content.replace(env_name, env_value)

    tmp_dir = os.path.join(log_dir, "lm_eval_configs")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, os.path.basename(file_path))

    # Write modified config to temp file
    with open(tmp_file, 'w', encoding='utf-8') as f:
        f.write(file_content)

    # Check if has custom utils.py in the same directory
    # Needed for GPQA task
    custom_utils_path = os.path.join(os.path.dirname(file_path), 'utils.py')
    if os.path.exists(custom_utils_path):
        # copy utils.py to temp directory
        shutil.copy(custom_utils_path, tmp_dir)

    # Return temp directory
    return tmp_dir


def build_worker_environment(worker_config, env_config, role, benchmark_mode,
                             nsys_on, profile_range, concurrency, gpu_ids):
    """Build complete environment dictionary for worker processes.

    Args:
        worker_config: Worker configuration dict
        env_config: Environment configuration dict
        role: Server role ("CTX" or "GEN")
        benchmark_mode: Benchmark mode string
        nsys_on: Whether nsys profiling is enabled
        profile_range: Profile range string (e.g., "10-30")
        concurrency: Concurrency level
        gpu_ids: List of GPU IDs assigned to this worker

    Returns:
        Dictionary of environment variables

    Note:
        CUDA_VISIBLE_DEVICES is NOT set here. It is passed as an argument to
        start_worker.sh and set per-rank based on SLURM_LOCALID.
    """
    env = {}

    # 1. Use gpu_ids to set CUDA_VISIBLE_DEVICES
    cuda_devices = ','.join(map(str, gpu_ids))
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices

    # 2. Parse user-defined worker env vars from config
    worker_env_var = env_config.get('worker_env_var', '')
    for var_string in worker_env_var.split():
        if '=' in var_string:
            key, val = var_string.split('=', 1)
            env[key] = val

    # 3. Add role-specific env vars (CTX or GEN)
    role_env_vars = {
        "CTX": env_config.get('ctx_worker_env_var', ''),
        "GEN": env_config.get('gen_worker_env_var', '')
    }
    role_specific_env_var = role_env_vars.get(role, '')
    for var_string in role_specific_env_var.split():
        if '=' in var_string:
            key, val = var_string.split('=', 1)
            env[key] = val

    # 4. Add mode-based env vars
    if benchmark_mode == "gen_only_no_context":
        env["TRTLLM_DISAGG_BENCHMARK_GEN_ONLY"] = "1"
    if benchmark_mode == "gen_only":
        env["TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP"] = "1"
        if role == "GEN":
            env["TLLM_BENCHMARK_REQ_QUEUES_SIZE"] = str(concurrency)

    # 5. Add profiling env vars (conditional)
    if nsys_on:
        env["TLLM_PROFILE_RECORD_GC"] = "1"
        env["TLLM_NVTX_DEBUG"] = "1"
        env["NSYS_MPI_STORE_TEAMS_PER_RANK"] = "1"
        env["TLLM_PROFILE_START_STOP"] = profile_range

    return env


def build_server_environment(env_config, benchmark_mode):
    """Build complete environment dictionary for server process.

    Args:
        env_config: Environment configuration dict
        benchmark_mode: Benchmark mode string

    Returns:
        Dictionary of environment variables
    """
    env = {}

    # Parse user-defined server env vars
    server_env_var = env_config.get('server_env_var', '')
    for var_string in server_env_var.split():
        if '=' in var_string:
            key, val = var_string.split('=', 1)
            env[key] = val

    # Add mode-based env vars
    if benchmark_mode == "gen_only_no_context":
        env["TRTLLM_DISAGG_BENCHMARK_GEN_ONLY"] = "1"

    return env


def format_export_string(env_dict):
    """Convert environment dictionary to srun --export format.

    Args:
        env_dict: Dictionary of environment variables

    Returns:
        String formatted for srun --export flag (e.g., "KEY1=val1,KEY2=val2")
        Returns "NONE" if no variables specified.

    Note:
        Values containing commas are quoted to avoid conflicts with srun's delimiter.
    """
    if not env_dict:
        return "NONE"

    export_list = []
    for k, v in env_dict.items():
        # srun cannot handle values that contain commas
        if ',' in v:
            export_list.append(f"'{k}={v}'")
        else:
            export_list.append(f"{k}={v}")
    return ",".join(export_list)


def save_env_file(env_file, server_env_var, worker_env_var, ctx_worker_env_var,
                  gen_worker_env_var):

    def get_env_var_str(env_var_str):
        env_data = {}
        for env_var in env_var_str.split():
            if '=' in env_var:
                key, value = env_var.split('=', 1)
                env_data[key] = value
        return env_data

    env_data = {}
    env_data['server_env_var'] = get_env_var_str(server_env_var)
    env_data['worker_env_var'] = get_env_var_str(worker_env_var)
    env_data['ctx_worker_env_var'] = get_env_var_str(ctx_worker_env_var)
    env_data['gen_worker_env_var'] = get_env_var_str(gen_worker_env_var)
    with open(env_file, 'w') as f:
        json.dump(env_data, f, indent=2)
    print(f"Environment variables saved to {env_file}")


def submit_job(config, log_dir, dry_run):
    # Extract configurations
    slurm_config = config['slurm']
    slurm_config.setdefault('extra_args', '')
    slurm_config.setdefault('set_segment', True)

    hw_config = config['hardware']
    env_config = config['environment']
    worker_config = config['worker_config']
    benchmark_config = config['benchmark']

    if 'work_dir' in env_config and os.path.isdir(env_config['work_dir']):
        script_dir = env_config['work_dir']
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))

    # Set default accuracy configuration for backward compatibility
    if 'accuracy' not in config:
        config['accuracy'] = {
            'enable_accuracy_test':
            False,
            'model':
            'local-completions',
            'tasks':
            'gsm8k',
            'model_args_extra':
            'num_concurrent=512,max_retries=3,tokenized_requests=false,timeout=1200,max_gen_toks=256,max_length=4096'
        }

    # Set default environment configuration for backward compatibility
    env_config.setdefault('trtllm_repo', '')
    env_config.setdefault('build_wheel', False)
    env_config.setdefault('cuda_architectures', '')
    env_config.setdefault('trtllm_wheel_path', '')
    env_config.setdefault('worker_env_var', '')
    env_config.setdefault('server_env_var', '')

    profiling_config = config.get('profiling', {})
    profiling_config.setdefault('nsys_on', False)
    profiling_config.setdefault('ctx_profile_range', '10-30')
    profiling_config.setdefault('gen_profile_range', '200-250')

    # Get number of servers from config
    ctx_num = hw_config['num_ctx_servers']
    gen_num = hw_config['num_gen_servers']
    gpus_per_node = hw_config['gpus_per_node']

    # Calculate nodes based on world sizes
    ctx_tp_size = worker_config['ctx'].get('tensor_parallel_size', 1)
    ctx_cp_size = worker_config['ctx'].get('context_parallel_size', 1)
    ctx_pp_size = worker_config['ctx'].get('pipeline_parallel_size', 1)
    ctx_world_size = ctx_tp_size * ctx_cp_size * ctx_pp_size
    ctx_nodes = calculate_nodes(ctx_world_size, ctx_num, gpus_per_node)

    gen_tp_size = worker_config['gen'].get('tensor_parallel_size', 1)
    gen_cp_size = worker_config['gen'].get('context_parallel_size', 1)
    gen_pp_size = worker_config['gen'].get('pipeline_parallel_size', 1)
    gen_world_size = gen_tp_size * gen_cp_size * gen_pp_size
    gen_nodes = calculate_nodes(gen_world_size, gen_num, gpus_per_node)
    ucx_warmup_requests = 2 * ctx_world_size * \
        gen_world_size if benchmark_config['mode'] == "e2e" else 0

    total_nodes = ctx_nodes + gen_nodes
    total_tasks = total_nodes * gpus_per_node

    # Generate log directory path based on configuration
    isl = benchmark_config['input_length']
    osl = benchmark_config['output_length']
    gen_batch_size = worker_config['gen']['max_batch_size']
    gen_enable_attention_dp = worker_config['gen']['enable_attention_dp']

    # Get eplb num_slots for gen worker
    load_balancer_config = worker_config['gen'].get('moe_config', {}).get(
        'load_balancer', {})
    if isinstance(load_balancer_config, str):
        with open(load_balancer_config, 'r') as f:
            load_balancer_config = yaml.safe_load(f)
    eplb_num_slots = load_balancer_config.get('num_slots', 0)

    # Get mtp_size from gen config's speculative_config
    mtp_size = worker_config['gen'].get('speculative_config',
                                        {}).get('num_nextn_predict_layers', 0)

    # Create base log directory path
    if 'log_dir' in env_config and env_config['log_dir']:
        log_dir = env_config['log_dir']
    if log_dir is None:
        log_base = os.path.join(script_dir, "logs")

        date_prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_base = os.path.join(log_base, f"{date_prefix}/{isl}-{osl}")

        # Determine directory suffix based on attention_dp
        if gen_enable_attention_dp:
            dir_suffix = f"disagg_ctx{ctx_num}_gen{gen_num}_dep{gen_tp_size}_batch{gen_batch_size}_eplb{eplb_num_slots}_mtp{mtp_size}"
        else:
            dir_suffix = f"disagg_ctx{ctx_num}_gen{gen_num}_tep{gen_tp_size}_batch{gen_batch_size}_eplb{eplb_num_slots}_mtp{mtp_size}"

        # Create full log directory path
        log_dir = os.path.join(log_base, dir_suffix)

    # if trtllm_config.yaml exists, don't remove the directory, remove other files in the directory except trtllm_config.yaml
    # also don't remove concurrency_* folders
    if os.path.exists(log_dir):
        if not os.path.exists(os.path.join(log_dir, 'trtllm_config.yaml')):
            print(f"[WARNING] Removing existing log directory: {log_dir}")
            shutil.rmtree(log_dir)
        else:
            print(
                f"[WARNING] trtllm_config.yaml exists, not removing the directory: {log_dir}"
            )
            for file in os.listdir(log_dir):
                if file != 'trtllm_config.yaml' and not file.startswith(
                        'concurrency_'):
                    if os.path.isdir(os.path.join(log_dir, file)):
                        shutil.rmtree(os.path.join(log_dir, file))
                    else:
                        os.remove(os.path.join(log_dir, file))
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log will be saved to: {log_dir}")

    # Save environment variables (for record-keeping only)
    worker_env_var = env_config.get('worker_env_var', '')
    ctx_worker_env_var = env_config.get('ctx_worker_env_var', '')
    gen_worker_env_var = env_config.get('gen_worker_env_var', '')
    server_env_var = env_config.get('server_env_var', '')
    save_env_file(os.path.join(log_dir, "env_vars.json"), server_env_var,
                  worker_env_var, ctx_worker_env_var, gen_worker_env_var)

    # Setup config file paths and save worker configs
    ctx_config_path = os.path.join(log_dir, 'ctx_config.yaml')
    gen_config_path = os.path.join(log_dir, 'gen_config.yaml')
    save_worker_config(worker_config['ctx'], ctx_config_path)
    save_worker_config(worker_config['gen'], gen_config_path)

    # Prepare allocation template
    allocations = allocate_gpus(
        total_nodes=total_nodes,
        gpus_per_node=gpus_per_node,
        num_gen_servers=gen_num,
        num_ctx_servers=ctx_num,
        gen_world_size=gen_world_size,
        ctx_world_size=ctx_world_size,
    )
    with open(os.path.join(log_dir, "allocations.json"), "w") as f:
        json.dump(allocations, f, indent=2)

    # Generate disagg server config
    server_config = convert_allocations_to_server_config(allocations)
    with open(os.path.join(log_dir, "server_config_base.yaml"), "w") as f:
        yaml.dump(server_config, f)
    disagg_server_hostname = server_config['hostname']
    disagg_server_port = server_config['port']

    container_name = "disaggr-test"
    start_server_cmds = []
    container_mount_str = env_config['container_mount']
    container_mount_str += f",{script_dir}:{script_dir}"

    # Pre-define server-type-specific configurations
    server_configs = {
        "GEN": {
            "world_size": gen_world_size,
            "profile_range": profiling_config['gen_profile_range'],
            "config_path": gen_config_path
        },
        "CTX": {
            "world_size": ctx_world_size,
            "profile_range": profiling_config['ctx_profile_range'],
            "config_path": ctx_config_path
        }
    }

    # Generate start worker commands with placeholder hostnames
    for server_type in allocations.keys():
        server_cfg = server_configs[server_type]

        for server_id in allocations[server_type].keys():
            allocation = allocations[server_type][server_id]
            # Get GPU IDs for this server from allocation
            # When multi-node, all nodes have same device list, so use first node [0]
            gpu_ids = list(allocation["nodes"].values())[0]

            # Build environment for this worker
            worker_env = build_worker_environment(
                worker_config=worker_config,
                env_config=env_config,
                role=server_type,
                benchmark_mode=benchmark_config['mode'],
                nsys_on=profiling_config['nsys_on'],
                profile_range=server_cfg['profile_range'],
                concurrency=benchmark_config['concurrency_list'].split(',')[0],
                gpu_ids=gpu_ids,
            )
            export_str = format_export_string(worker_env)

            # Use script_dir for start_worker.sh
            cmd = [
                "srun -l",
                f"--nodelist {','.join(allocation['nodes'].keys())}",
                f"-N {len(allocation['nodes'])}",
                f"--ntasks {server_cfg['world_size']}",
                f"--ntasks-per-node {gpus_per_node}",
                f"--export=\"{export_str}\"",
                f"--container-image {env_config['container_image']}",
                f"--container-name {container_name}",
                f"--container-mounts {container_mount_str}",
                "--no-container-mount-home --mpi=pmix --overlap",
                f"bash {os.path.join(script_dir, 'start_worker.sh')}",
                server_type,
                str(server_id),
                env_config['model_path'],
                str(allocation["port"]),
                str(slurm_config['numa_bind']).lower(),
                log_dir,
                str(profiling_config['nsys_on']).lower(),
                server_cfg['config_path'],
                f"&> {log_dir}/3_output_{server_type}_{server_id}.log &",
            ]
            start_server_cmds.append(" ".join(cmd))

    # Generate start server commands (use script_dir for start_server.sh)
    server_env = build_server_environment(env_config, benchmark_config['mode'])
    export_str = format_export_string(server_env)

    cmd = [
        "srun -l",
        f"--nodelist {disagg_server_hostname}",
        f"--container-name={container_name}",
        f"--export=\"{export_str}\"",
        f"--container-image={env_config['container_image']}",
        f"--container-mounts={container_mount_str}",
        f"--no-container-mount-home --mpi=pmix --overlap -N 1 -n 1",
        f"bash {os.path.join(script_dir, 'start_server.sh')} {os.path.join(log_dir, 'server_config.yaml')}",
        f"&> {log_dir}/4_output_server.log &",
    ]
    start_server_cmds.append(" ".join(cmd))

    # Generate wait server command (use script_dir for wait_server.sh)
    cmd = [
        "srun -l",
        f"--container-name={container_name}",
        f"--container-mounts={container_mount_str}",
        f"--mpi=pmix --overlap -N 1 -n 1",
        f"bash {os.path.join(script_dir, 'wait_server.sh')} {disagg_server_hostname} {disagg_server_port}",
        f"&> {log_dir}/5_wait_server.log",
    ]
    start_server_cmds.append(" ".join(cmd))

    with open(os.path.join(log_dir, "start_server_cmds_base.sh"), "w") as f:
        f.write("\n".join(start_server_cmds) + "\n")

    # Generate client commands (use script_dir for benchmark scripts)
    client_cmds = []
    client_slurm_prefix = [
        f"srun -l --container-name={container_name}",
        f"--container-mounts={container_mount_str}",
        f"--mpi=pmix --overlap -N 1 -n 1",
    ]
    # Append benchmark commands
    if benchmark_config.get('enable_benchmark', True):
        env_var = config['benchmark'].get('env_var', {})
        benchmark_prefix = client_slurm_prefix + [
            f"--export \"{convert_envs_to_str(env_var)}\""
        ]
        if benchmark_config['use_nv_sa_benchmark']:
            if benchmark_config['mode'] == "gen_only":
                print(
                    f"[ERROR] SA benchmark client script is not supported for gen_only mode"
                )
                sys.exit(1)
            benchmark_cmd = [
                f"bash {os.path.join(script_dir, 'run_benchmark_nv_sa.sh')}",
                f"'{env_config['model_path']}' {isl} {osl} {benchmark_config['benchmark_ratio']} {benchmark_config['multi_round']} {gen_num} '{benchmark_config['concurrency_list']}' {benchmark_config['streaming']} '{log_dir}' {disagg_server_hostname} {disagg_server_port} {ucx_warmup_requests}",
                f"&> {log_dir}/6_bench.log"
            ]
            client_cmds.append(" ".join(benchmark_prefix + benchmark_cmd))
        else:
            benchmark_cmd = [
                f"bash {os.path.join(script_dir, 'run_benchmark.sh')}",
                f"'{env_config['model_path']}' '{benchmark_config['dataset_file']}' {benchmark_config['multi_round']} {gen_num} '{benchmark_config['concurrency_list']}' {benchmark_config['streaming']} '{log_dir}' {disagg_server_hostname} {disagg_server_port} {ucx_warmup_requests}",
                f"&> {log_dir}/6_bench.log"
            ]
            client_cmds.append(" ".join(benchmark_prefix + benchmark_cmd))

    # Append accuracy test commands
    if config['accuracy']['enable_accuracy_test']:
        env_var = config['accuracy'].get('env_var', {})
        accuracy_prefix = client_slurm_prefix + [
            f"--export \"{convert_envs_to_str(env_var)}\""
        ]
        for task in config['accuracy']['tasks']:
            extra_kwargs = config['accuracy']['tasks'][task].get(
                'extra_kwargs', {})
            extra_kwargs_str = ""
            for key, value in extra_kwargs.items():
                if isinstance(value, bool):
                    if value:
                        extra_kwargs_str += f" --{key}"
                elif key == "custom_config":
                    extra_kwargs_str += f" --include_path={replace_env_in_file(log_dir, value, env_var)}"
                else:
                    extra_kwargs_str += f" --{key}='{value}'"
            end_point_map = {
                'local-completions': 'v1/completions',
                'local-chat-completions': 'v1/chat/completions',
            }
            model = config['accuracy']['tasks'][task]['model']
            accuracy_cmd = [
                'lm_eval', '--model', model, '--tasks', task, '--model_args',
                f"model={env_config['model_path']},base_url=http://{disagg_server_hostname}:{disagg_server_port}/{end_point_map[model]},{config['accuracy']['tasks'][task]['model_args_extra']}",
                '--log_samples', '--output_path',
                f'{log_dir}/accuracy_eval_{task}', extra_kwargs_str,
                f"&> {log_dir}/7_accuracy_eval_{task}.log"
            ]
            client_cmds.append(" ".join(accuracy_prefix + accuracy_cmd))

    # record ${SLURM_JOB_NODELIST} to ${log_dir}/8_done_job_id.txt
    done_cmd = [
        "echo", "${SLURM_JOB_NODELIST}", ">",
        f"{log_dir}/8_done_${{SLURM_JOB_ID}}.txt"
    ]
    client_cmds.append(" ".join(done_cmd))

    with open(os.path.join(log_dir, "client_cmds_base.sh"), "w") as f:
        f.write("\n".join(client_cmds) + "\n")

    # Resolve slurm script_file path
    # If it's a relative path, make it relative to script_dir
    slurm_script_file = slurm_config['script_file']
    if not os.path.isabs(slurm_script_file):
        slurm_script_file = os.path.join(script_dir, slurm_script_file)

    # Verify the script file exists
    if not os.path.exists(slurm_script_file):
        print(f"[ERROR] SLURM script file not found: {slurm_script_file}",
              file=sys.stderr)
        sys.exit(1)

    # Prepare sbatch command
    # yapf: disable
    cmd = [
        'sbatch',
        f'--partition={slurm_config["partition"]}',
        f'--account={slurm_config["account"]}',
        f'--time={slurm_config["job_time"]}',
        f'--job-name={slurm_config["job_name"]}',
        f'--nodes={total_nodes}',
        f'--ntasks={total_tasks}',
        f'--ntasks-per-node={hw_config["gpus_per_node"]}',
        *([] if not slurm_config['set_segment']
          else [f'--segment={total_nodes}']),
        f'--output={log_dir}/slurm-%j.out',
        f'--error={log_dir}/slurm-%j.err',
        *([arg for arg in slurm_config['extra_args'].split() if arg]),
        slurm_script_file,

        # Benchmark Configuration
        '--benchmark-mode', benchmark_config['mode'],

        # Environment and paths
        '--trtllm-repo', env_config['trtllm_repo'],
        '--work-dir', script_dir,
        '--full-logdir', log_dir,
        '--container-name', container_name,
        '--container-mount', container_mount_str,
        '--container-image', env_config['container_image'],
        '--build-wheel', str(env_config['build_wheel']).lower(),
        '--cuda-architectures', env_config['cuda_architectures'],
        '--trtllm-wheel-path', env_config['trtllm_wheel_path'],
    ]
    # yapf: enable

    if dry_run:
        print(
            "[WARNING] Dry run mode, will not submit the job. This should be used for test purpose only."
        )
        print("sbatch command:")
        print(" ".join(cmd))
        return
    else:
        # Submit the job
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job: {e}", file=sys.stderr)
            sys.exit(1)


def main():
    args = parse_args()

    # Determine which mode to use
    if args.config:
        # Single config file mode
        config_files = [args.config]
    else:
        # Directory mode - find all YAML files
        yaml_pattern = os.path.join(args.dir, '*.yaml')
        config_files = sorted(glob.glob(yaml_pattern))

        if not config_files:
            print(f"No YAML files found in directory: {args.dir}",
                  file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(config_files)} YAML file(s) in {args.dir}")

    # Process each config file
    for config_file in config_files:
        print(f"Processing: {config_file}")
        try:
            config = load_config(config_file)
            submit_job(config, args.log_dir, args.dry_run)
            print(f"Successfully submitted job for: {config_file}\n")
        except Exception as e:
            traceback.print_exc()
            print(f"Error processing {config_file}: {e}", file=sys.stderr)
            # Continue processing other files even if one fails
            continue


if __name__ == '__main__':
    main()
