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
        if server_type == "GEN":
            generation_servers = {'num_instances': num_servers, 'urls': urls}
            server_hostname = urls[0].split(':')[0]
            if allocations[server_type][server_id]['port'] == server_port:
                server_port += 1  # Avoid port conflict
        elif server_type == "CTX":
            context_servers = {'num_instances': num_servers, 'urls': urls}

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


def submit_job(config, log_dir, dry_run):
    # Extract configurations
    slurm_config = config['slurm']
    slurm_config.setdefault('extra_args', '')
    slurm_config.setdefault('set_segment', True)

    hw_config = config['hardware']
    env_config = config['environment']
    worker_config = config['worker_config']
    benchmark_config = config['benchmark']

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

    worker_env_var = env_config.get('worker_env_var')
    server_env_var = env_config.get('server_env_var')
    if benchmark_config['mode'] == "gen_only_no_context":
        worker_env_var += " TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1"
        server_env_var += " TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1"

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
    if log_dir is None:
        log_base = os.path.join(env_config['work_dir'], "logs")

        date_prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_base = os.path.join(log_base, f"{date_prefix}/{isl}-{osl}")

        # Determine directory suffix based on attention_dp
        if gen_enable_attention_dp:
            dir_suffix = f"disagg_ctx{ctx_num}_gen{gen_num}_dep{gen_tp_size}_batch{gen_batch_size}_eplb{eplb_num_slots}_mtp{mtp_size}"
        else:
            dir_suffix = f"disagg_ctx{ctx_num}_gen{gen_num}_tep{gen_tp_size}_batch{gen_batch_size}_eplb{eplb_num_slots}_mtp{mtp_size}"

        # Create full log directory path
        log_dir = os.path.join(log_base, dir_suffix)

    # Remove existing directory if it exists
    if os.path.exists(log_dir):
        print(f"[WARNING] Removing existing log directory: {log_dir}")
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    print(f"Log will be saved to: {log_dir}")

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
    with open(os.path.join(log_dir, "server_config.yaml"), "w") as f:
        yaml.dump(server_config, f)
    disagg_server_hostname = server_config['hostname']
    disagg_server_port = server_config['port']

    container_name = "disaggr-test"
    start_server_cmds = []
    # Generate start worker commands with placeholder hostnames
    for server_type in allocations.keys():
        for server_id in allocations[server_type].keys():
            allocation = allocations[server_type][server_id]
            cuda_devices = ",".join([
                str(device) for device in list(allocation["nodes"].values())[0]
            ])
            cur_worker_env_var = worker_env_var + f" CUDA_VISIBLE_DEVICES={cuda_devices}"
            cmd = [
                "srun -l",
                f"--nodelist {','.join(allocation['nodes'].keys())}",
                f"-N {len(allocation['nodes'])}",
                f"--ntasks {gen_world_size if server_type == 'GEN' else ctx_world_size}",
                f"--ntasks-per-node {gpus_per_node}",
                f"--container-image {env_config['container_image']}",
                f"--container-name {container_name}",
                f"--container-mounts {env_config['container_mount']}",
                "--no-container-mount-home --mpi=pmix --overlap",
                f"bash {os.path.join(env_config['work_dir'], 'start_worker.sh')}",
                server_type,
                str(server_id),
                env_config['model_path'],
                str(allocation["port"]),
                benchmark_config['mode'],
                f"'{benchmark_config['concurrency_list']}'",
                str(slurm_config['numa_bind']).lower(),
                log_dir,
                str(profiling_config['nsys_on']).lower(),
                f"'{profiling_config['gen_profile_range']}'" if server_type
                == "GEN" else f"'{profiling_config['ctx_profile_range']}'",
                gen_config_path if server_type == "GEN" else ctx_config_path,
                f"'{cur_worker_env_var}'",
                f"&> {log_dir}/3_output_{server_type}_{server_id}.log &",
            ]
            start_server_cmds.append(" ".join(cmd))

    # Generate start server commands
    cmd = [
        "srun -l",
        f"--nodelist {disagg_server_hostname}",
        f"--container-name={container_name}",
        f"--container-image={env_config['container_image']}",
        f"--container-mounts={env_config['container_mount']}",
        f"--no-container-mount-home --mpi=pmix --overlap -N 1 -n 1",
        f"bash {env_config['work_dir']}/start_server.sh {os.path.join(log_dir, 'server_config.yaml')} \"{server_env_var}\"",
        f"&> {log_dir}/4_output_server.log &",
    ]
    start_server_cmds.append(" ".join(cmd))

    # Generate wait server command
    cmd = [
        "srun -l",
        f"--container-name={container_name}",
        f"--container-mounts={env_config['container_mount']}",
        f"--mpi=pmix --overlap -N 1 -n 1",
        f"bash {env_config['work_dir']}/wait_server.sh {disagg_server_hostname} {disagg_server_port}",
        f"&> {log_dir}/5_wait_server.log",
    ]
    start_server_cmds.append(" ".join(cmd))

    with open(os.path.join(log_dir, "start_server_cmds.sh"), "w") as f:
        f.write("\n".join(start_server_cmds) + "\n")

    # Generate client commands
    client_cmds = []
    client_slurm_prefix = [
        f"srun -l --container-name={container_name}",
        f"--container-mounts={env_config['container_mount']}",
        f"--mpi=pmix --overlap -N 1 -n 1",
    ]

    # Append benchmark commands
    if benchmark_config.get('enable_benchmark', True):
        env_var = config['benchmark'].get('env_var', {})
        benchmark_prefix = client_slurm_prefix + [
            f"--export \"{convert_envs_to_str(env_var)}\""
        ]
        if benchmark_config['use_nv_sa_benchmark']:
            benchmark_cmd = [
                f"bash {env_config['work_dir']}/run_benchmark_nv_sa.sh",
                f"'{env_config['model_path']}' {isl} {osl} {benchmark_config['benchmark_ratio']} {benchmark_config['multi_round']} {gen_num} '{benchmark_config['concurrency_list']}' {benchmark_config['streaming']} '{log_dir}' {disagg_server_hostname} {disagg_server_port}",
                f"&> {log_dir}/6_bench.log"
            ]
            client_cmds.append(" ".join(benchmark_prefix + benchmark_cmd))
        else:
            benchmark_cmd = [
                f"bash {env_config['work_dir']}/run_benchmark.sh",
                f"'{env_config['model_path']}' '{benchmark_config['dataset_file']}' {benchmark_config['multi_round']} {gen_num} '{benchmark_config['concurrency_list']}' {benchmark_config['streaming']} '{log_dir}' {disagg_server_hostname} {disagg_server_port}",
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
    with open(os.path.join(log_dir, "client_cmds.sh"), "w") as f:
        f.write("\n".join(client_cmds) + "\n")

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
        *([] if not slurm_config['set_segment'] else [f'--segment={total_nodes}']),
        f'--output={log_dir}/slurm-%j.out',
        f'--error={log_dir}/slurm-%j.err',
        *([arg for arg in slurm_config['extra_args'].split() if arg]),
        slurm_config['script_file'],

        # Benchmark Configuration
        '--benchmark-mode', benchmark_config['mode'],

        # Environment and paths
        '--trtllm-repo', env_config['trtllm_repo'],
        '--work-dir', env_config['work_dir'],
        '--full-logdir', log_dir,
        '--container-name', container_name,
        '--container-mount', env_config['container_mount'],
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
