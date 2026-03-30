#!/usr/bin/env python3
"""Submit DWDP disaggregated benchmark jobs.

This script handles the DWDP-specific submission flow which requires MPI-based
worker launching via ``trtllm-serve disaggregated_mpi_worker``.  It reuses
shared utilities from ``submit.py`` for config parsing, GPU allocation, and
sbatch command construction.
"""

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

from submit import (
    allocate_gpus,
    build_server_environment,
    build_worker_environment,
    calculate_nodes,
    convert_allocations_to_server_config,
    convert_envs_to_str,
    format_export_string,
    load_config,
    replace_env_in_file,
    save_env_file,
    save_worker_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Submit DWDP disaggregated benchmark job')
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


def generate_mpi_worker_config(worker_config, allocations, env_config,
                               disagg_hostname, disagg_port, output_path):
    """Generate a config YAML compatible with ``trtllm-serve disaggregated_mpi_worker``."""

    def _build_urls(server_type):
        urls = []
        for server_id in sorted(allocations.get(server_type, {}).keys()):
            inst = allocations[server_type][server_id]
            host = list(inst["nodes"].keys())[0]
            urls.append(f"{host}:{inst['port']}")
        return urls

    ctx_urls = _build_urls("CTX")
    gen_urls = _build_urls("GEN")

    ctx_section = dict(worker_config['ctx'])
    ctx_section['num_instances'] = len(ctx_urls)
    ctx_section['urls'] = ctx_urls

    gen_section = dict(worker_config['gen'])
    gen_section['num_instances'] = len(gen_urls)
    gen_section['urls'] = gen_urls

    config = {
        'model': env_config['model_path'],
        'hostname': disagg_hostname,
        'port': disagg_port,
        'backend': 'pytorch',
        'max_retries': 100,
        'context_servers': ctx_section,
        'generation_servers': gen_section,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def submit_dwdp_job(config, log_dir, dry_run):
    """Submit a DWDP disaggregated benchmark job."""
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

    ctx_num = hw_config['num_ctx_servers']
    gen_num = hw_config['num_gen_servers']
    gpus_per_node = hw_config['gpus_per_node']

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

    dwdp_size = worker_config.get('ctx', {}).get('dwdp_config',
                                                 {}).get('dwdp_size', 1)

    isl = benchmark_config['input_length']
    osl = benchmark_config['output_length']
    gen_batch_size = worker_config['gen']['max_batch_size']

    load_balancer_config = worker_config['gen'].get('moe_config', {}).get(
        'load_balancer', {})
    if isinstance(load_balancer_config, str):
        with open(load_balancer_config, 'r') as f:
            load_balancer_config = yaml.safe_load(f)
    eplb_num_slots = load_balancer_config.get('num_slots', 0)

    mtp_size = worker_config['gen'].get('speculative_config',
                                        {}).get('num_nextn_predict_layers', 0)

    if 'log_dir' in env_config and env_config['log_dir']:
        log_dir = env_config['log_dir']
    if log_dir is None:
        log_base = os.path.join(script_dir, "logs")

        date_prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_base = os.path.join(log_base, f"{date_prefix}/{isl}-{osl}")

        dir_suffix = f"disagg_ctx{ctx_num}_dwdp{dwdp_size}_gen{gen_num}_dep{gen_tp_size}_batch{gen_batch_size}_eplb{eplb_num_slots}_mtp{mtp_size}"

        log_dir = os.path.join(log_base, dir_suffix)

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

    ctx_config_path = os.path.join(log_dir, 'ctx_config.yaml')
    gen_config_path = os.path.join(log_dir, 'gen_config.yaml')
    save_worker_config(worker_config['ctx'], ctx_config_path)
    save_worker_config(worker_config['gen'], gen_config_path)

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

    server_config = convert_allocations_to_server_config(allocations)
    with open(os.path.join(log_dir, "server_config_base.yaml"), "w") as f:
        yaml.dump(server_config, f)
    disagg_server_hostname = server_config['hostname']
    disagg_server_port = server_config['port']

    container_name = "disaggr-test"
    start_server_cmds = []
    container_mount_str = env_config['container_mount']
    container_mount_str += f",{script_dir}:{script_dir}"

    # --- DWDP mode: single srun with disaggregated_mpi_worker ---
    mpi_config_base_path = os.path.join(log_dir,
                                        'mpi_worker_config_base.yaml')
    mpi_config_path = os.path.join(log_dir, 'mpi_worker_config.yaml')
    generate_mpi_worker_config(worker_config, allocations, env_config,
                               disagg_server_hostname, disagg_server_port,
                               mpi_config_base_path)

    ctx_node_list = []
    for sid in sorted(allocations.get("CTX", {}).keys()):
        for node in allocations["CTX"][sid]["nodes"]:
            if node not in ctx_node_list:
                ctx_node_list.append(node)
    gen_node_list = []
    for sid in sorted(allocations.get("GEN", {}).keys()):
        for node in allocations["GEN"][sid]["nodes"]:
            if node not in gen_node_list:
                gen_node_list.append(node)
    mpi_nodelist = ctx_node_list + gen_node_list
    total_mpi_tasks = ctx_num * ctx_world_size + gen_num * gen_world_size
    mpi_num_nodes = len(mpi_nodelist)
    num_ctx_gpus = ctx_num * ctx_world_size
    worker_env_var = env_config.get('worker_env_var', '')
    ctx_worker_env_var = env_config.get('ctx_worker_env_var', '')
    gen_worker_env_var = env_config.get('gen_worker_env_var', '')
    dwdp_ctx_worker_env_var = worker_env_var + \
        (f" {ctx_worker_env_var}" if ctx_worker_env_var else "")
    dwdp_gen_worker_env_var = worker_env_var + \
        (f" {gen_worker_env_var}" if gen_worker_env_var else "")

    cmd = [
        "srun -l",
        f"--nodelist {','.join(mpi_nodelist)}",
        f"-N {mpi_num_nodes}",
        f"--ntasks {total_mpi_tasks}",
        f"--ntasks-per-node {gpus_per_node}",
        f"--container-image {env_config['container_image']}",
        f"--container-name {container_name}",
        f"--container-mounts {container_mount_str}",
        "--no-container-mount-home --mpi=pmix --overlap",
        f"bash {os.path.join(script_dir, 'start_worker_dwdp.sh')}",
        mpi_config_path,
        str(slurm_config['numa_bind']).lower(),
        log_dir,
        str(profiling_config['nsys_on']).lower(),
        f"'{profiling_config['ctx_profile_range']}'",
        f"'{profiling_config['gen_profile_range']}'",
        str(num_ctx_gpus),
        f"'{dwdp_ctx_worker_env_var}'",
        f"'{dwdp_gen_worker_env_var}'",
        f"&> {log_dir}/3_output_workers.log &",
    ]
    start_server_cmds.append(" ".join(cmd))

    # Generate start server commands
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

    save_env_file(
        os.path.join(log_dir, "env_vars.json"),
        env_config.get('server_env_var', ''),
        env_config.get('worker_env_var', ''),
        env_config.get('ctx_worker_env_var', ''),
        env_config.get('gen_worker_env_var', ''),
    )

    # Generate wait server command
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

    # Generate client commands
    client_cmds = []
    client_slurm_prefix = [
        f"srun -l --container-name={container_name}",
        f"--container-mounts={container_mount_str}",
        f"--mpi=pmix --overlap -N 1 -n 1",
    ]
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

    done_cmd = [
        "echo", "${SLURM_JOB_NODELIST}", ">",
        f"{log_dir}/8_done_${{SLURM_JOB_ID}}.txt"
    ]
    client_cmds.append(" ".join(done_cmd))

    with open(os.path.join(log_dir, "client_cmds_base.sh"), "w") as f:
        f.write("\n".join(client_cmds) + "\n")

    slurm_script_file = slurm_config['script_file']
    if not os.path.isabs(slurm_script_file):
        slurm_script_file = os.path.join(script_dir, slurm_script_file)

    if not os.path.exists(slurm_script_file):
        print(f"[ERROR] SLURM script file not found: {slurm_script_file}",
              file=sys.stderr)
        sys.exit(1)

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

        '--benchmark-mode', benchmark_config['mode'],

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
        yaml_pattern = os.path.join(args.dir, '*.yaml')
        config_files = sorted(glob.glob(yaml_pattern))

        if not config_files:
            print(f"No YAML files found in directory: {args.dir}",
                  file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(config_files)} YAML file(s) in {args.dir}")

    for config_file in config_files:
        print(f"Processing: {config_file}")
        try:
            config = load_config(config_file)
            submit_dwdp_job(config, args.log_dir, args.dry_run)
            print(f"Successfully submitted job for: {config_file}\n")
        except Exception as e:
            traceback.print_exc()
            print(f"Error processing {config_file}: {e}", file=sys.stderr)
            continue


if __name__ == '__main__':
    main()
