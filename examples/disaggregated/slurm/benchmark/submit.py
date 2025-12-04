#!/usr/bin/env python3

import argparse
import glob
import os
import shutil
import subprocess
import sys
from datetime import datetime

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
    group.add_argument('--log-dir',
                       type=str,
                       default=None,
                       help='Log directory')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_worker_config(config, output_path, worker_type):
    """Save worker config to a separate YAML file."""
    # Get just the worker configuration without the wrapper
    worker_config = config['worker_config'][worker_type]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(worker_config, f, default_flow_style=False)


def calculate_nodes(world_size, num_servers, gpus_per_node):
    """Calculate required nodes based on world size and server count."""
    return (world_size + gpus_per_node - 1) // gpus_per_node * num_servers


def submit_job(config, log_dir):
    # Extract configurations
    slurm_config = config['slurm']
    slurm_config.setdefault('extra_args', '')

    hw_config = config['hardware']
    env_config = config['environment']

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

    # Get mtp_size from gen config's speculative_config
    gen_config = config['worker_config']['gen']
    mtp_size = gen_config.get('speculative_config',
                              {}).get('num_nextn_predict_layers', 0)

    # Calculate nodes based on world sizes
    ctx_tp_size = config['worker_config']['ctx']['tensor_parallel_size']
    ctx_cp_size = config['worker_config']['ctx']['context_parallel_size']
    ctx_pp_size = config['worker_config']['ctx']['pipeline_parallel_size']
    ctx_world_size = ctx_tp_size * ctx_cp_size * ctx_pp_size
    ctx_nodes = calculate_nodes(ctx_world_size, ctx_num,
                                hw_config['gpus_per_node'])
    gen_tp_size = config['worker_config']['gen']['tensor_parallel_size']
    gen_cp_size = config['worker_config']['gen']['context_parallel_size']
    gen_pp_size = config['worker_config']['gen']['pipeline_parallel_size']
    gen_world_size = gen_tp_size * gen_cp_size * gen_pp_size
    gen_nodes = calculate_nodes(gen_world_size, gen_num,
                                hw_config['gpus_per_node'])
    total_nodes = ctx_nodes + gen_nodes
    total_tasks = total_nodes * hw_config['gpus_per_node']

    # Generate log directory path based on configuration
    isl = config['benchmark']['input_length']
    osl = config['benchmark']['output_length']
    gen_batch_size = config['worker_config']['gen']['max_batch_size']
    gen_enable_attention_dp = config['worker_config']['gen'][
        'enable_attention_dp']

    if log_dir is None:
        # Create base log directory path
        date_prefix = datetime.now().strftime("%Y%m%d")
        log_base = os.path.join(env_config['work_dir'],
                                f"{date_prefix}/{isl}-{osl}")

        # Get eplb num_slots for gen worker
        load_balancer_config = config['worker_config']['gen'].get(
            'moe_config', {}).get('load_balancer', {})
        if isinstance(load_balancer_config, str):
            with open(load_balancer_config, 'r') as f:
                load_balancer_config = yaml.safe_load(f)
        eplb_num_slots = load_balancer_config.get('num_slots', 0)

        # Determine directory suffix based on attention_dp
        if gen_enable_attention_dp:
            dir_suffix = f"ctx{ctx_num}_gen{gen_num}_dep{gen_tp_size}_batch{gen_batch_size}_eplb{eplb_num_slots}_mtp{mtp_size}"
        else:
            dir_suffix = f"ctx{ctx_num}_gen{gen_num}_tep{gen_tp_size}_batch{gen_batch_size}_eplb{eplb_num_slots}_mtp{mtp_size}"

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
    save_worker_config(config, ctx_config_path, 'ctx')
    save_worker_config(config, gen_config_path, 'gen')

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
        f'--segment={total_nodes}',
        f'--gpus-per-node={hw_config["gpus_per_node"]}',
        *([arg for arg in slurm_config['extra_args'].split() if arg]),
        slurm_config['script_file'],
        # Hardware configuration
        '--gpus-per-node', str(hw_config['gpus_per_node']),
        '--numa-bind', str(slurm_config['numa_bind']).lower(),
        '--ctx-nodes', str(ctx_nodes),  # Number of nodes needed for ctx workers
        '--gen-nodes', str(gen_nodes),  # Number of nodes needed for gen workers
        '--ctx-world-size', str(ctx_world_size),  # World size for ctx workers
        '--gen-world-size', str(gen_world_size),  # World size for gen workers

        # Worker configuration
        '--num-ctx-servers', str(ctx_num),
        '--ctx-config-path', ctx_config_path,
        '--num-gen-servers', str(gen_num),
        '--gen-config-path', gen_config_path,
        '--concurrency-list', config['benchmark']['concurrency_list'],

        # Sequence and benchmark parameters
        '--isl', str(config['benchmark']['input_length']),
        '--osl', str(config['benchmark']['output_length']),
        '--multi-round', str(config['benchmark']['multi_round']),
        '--benchmark-ratio', str(config['benchmark']['benchmark_ratio']),
        '--streaming', str(config['benchmark']['streaming']).lower(),
        '--use-nv-sa-benchmark', str(config['benchmark']['use_nv_sa_benchmark']).lower(),
        '--benchmark-mode', config['benchmark']['mode'],
        '--cache-max-tokens', str(config['worker_config']['gen']['cache_transceiver_config']
            ['max_tokens_in_buffer']),

        # Environment and paths
        '--dataset-file', config['benchmark']['dataset_file'],
        '--model-path', env_config['model_path'],
        '--trtllm-repo', env_config['trtllm_repo'],
        '--work-dir', env_config['work_dir'],
        '--full-logdir', log_dir,
        '--container-mount', env_config['container_mount'],
        '--container-image', env_config['container_image'],
        '--build-wheel', str(env_config['build_wheel']).lower(),
        '--cuda-architectures', env_config['cuda_architectures'],
        '--trtllm-wheel-path', env_config['trtllm_wheel_path'],

        # Profiling
        '--nsys-on', str(profiling_config['nsys_on']).lower(),
        '--ctx-profile-range', profiling_config['ctx_profile_range'],
        '--gen-profile-range', profiling_config['gen_profile_range'],

        # Accuracy evaluation
        '--enable-accuracy-test', str(config['accuracy']['enable_accuracy_test']).lower(),
        '--accuracy-model', config['accuracy']['model'],
        '--accuracy-tasks', config['accuracy']['tasks'],
        '--model-args-extra', config['accuracy']['model_args_extra'],

        # Worker environment variables
        '--worker-env-var', env_config['worker_env_var'],

        # Server environment variables
        '--server-env-var', env_config['server_env_var']
    ]
    # yapf: enable

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
            submit_job(config, args.log_dir)
            print(f"Successfully submitted job for: {config_file}\n")
        except Exception as e:
            print(f"Error processing {config_file}: {e}", file=sys.stderr)
            # Continue processing other files even if one fails
            continue


if __name__ == '__main__':
    main()
