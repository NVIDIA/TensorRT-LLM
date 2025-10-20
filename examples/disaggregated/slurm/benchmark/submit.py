#!/usr/bin/env python3

import argparse
import glob
import os
import subprocess
import sys

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


def calculate_nodes(tp_size, num_servers, gpus_per_node):
    """Calculate required nodes based on tensor parallel size and server count."""
    return (tp_size + gpus_per_node - 1) // gpus_per_node * num_servers


def submit_job(config):
    # Extract configurations
    slurm_config = config['slurm']
    hw_config = config['hardware']
    env_config = config['environment']

    # Calculate nodes based on tensor parallel sizes
    ctx_tp_size = config['worker_config']['ctx']['tensor_parallel_size']
    gen_tp_size = config['worker_config']['gen']['tensor_parallel_size']

    # Get number of servers from config
    ctx_num = hw_config['num_ctx_servers']
    gen_num = hw_config['num_gen_servers']

    # Get mtp_size from gen config's speculative_config
    gen_config = config['worker_config']['gen']
    mtp_size = gen_config.get('speculative_config',
                              {}).get('num_nextn_predict_layers', 0)

    ctx_nodes = calculate_nodes(ctx_tp_size, ctx_num,
                                hw_config['gpus_per_node'])
    gen_nodes = calculate_nodes(gen_tp_size, gen_num,
                                hw_config['gpus_per_node'])
    total_nodes = ctx_nodes + gen_nodes
    total_tasks = total_nodes * hw_config['gpus_per_node']

    # Generate log directory path based on configuration
    isl = config['sequence']['input_length']
    osl = config['sequence']['output_length']
    gen_batch_size = config['worker_config']['gen']['max_batch_size']
    gen_enable_attention_dp = config['worker_config']['gen'][
        'enable_attention_dp']

    # Create base log directory path
    log_base = os.path.join(env_config['work_dir'], f"{isl}-{osl}")

    # Determine directory suffix based on attention_dp
    if gen_enable_attention_dp:
        dir_suffix = f"ctx{ctx_num}_gen{gen_num}_dep{gen_tp_size}_batch{gen_batch_size}_eplb{config['worker_config']['eplb_num_slots']}_mtp{mtp_size}"
    else:
        dir_suffix = f"ctx{ctx_num}_gen{gen_num}_tep{gen_tp_size}_batch{gen_batch_size}_eplb{config['worker_config']['eplb_num_slots']}_mtp{mtp_size}"

    # Create full log directory path
    log_dir = os.path.join(log_base, dir_suffix)
    os.makedirs(log_dir, exist_ok=True)

    # Setup config file paths and save worker configs
    ctx_config_path = os.path.join(log_dir, 'ctx_config.yaml')
    gen_config_path = os.path.join(log_dir, 'gen_config.yaml')
    save_worker_config(config, ctx_config_path, 'ctx')
    save_worker_config(config, gen_config_path, 'gen')

    # Prepare sbatch command
    cmd = [
        'sbatch',
        f'--partition={slurm_config["partition"]}',
        f'--gres=gpu:{hw_config["gpus_per_node"]}',
        f'--account={slurm_config["account"]}',
        f'--time={slurm_config["job_time"]}',
        f'--job-name={slurm_config["job_name"]}',
        f'--nodes={total_nodes}',
        f'--ntasks={total_tasks}',
        f'--ntasks-per-node={hw_config["gpus_per_node"]}',
        f'--segment={total_nodes}',
        slurm_config['script_file'],
        # Hardware configuration
        str(hw_config['gpus_per_node']),
        str(slurm_config['numa_bind']).lower(),
        str(ctx_nodes),  # Number of nodes needed for ctx workers
        str(gen_nodes),  # Number of nodes needed for gen workers
        str(ctx_tp_size),  # Tensor parallel size for ctx workers
        str(gen_tp_size),  # Tensor parallel size for gen workers

        # Worker configuration
        str(ctx_num),
        ctx_config_path,
        str(gen_num),
        gen_config_path,
        config['benchmark']['concurrency_list'],

        # Sequence and benchmark parameters
        str(config['sequence']['input_length']),
        str(config['sequence']['output_length']),
        str(config['benchmark']['multi_round']),
        str(config['benchmark']['benchmark_ratio']),
        str(config['benchmark']['streaming']).lower(),
        str(config['benchmark']['use_nv_sa_benchmark']).lower(),
        config['benchmark']['mode'],
        str(config['worker_config']['gen']['cache_transceiver_config']
            ['max_tokens_in_buffer']),

        # Environment and paths
        env_config['dataset_file'],
        env_config['model_path'],
        env_config['trtllm_repo'],
        env_config['work_dir'],
        log_dir,  # Pass the generated log directory
        env_config['container_mount'],
        env_config['container_image'],
        str(env_config['build_wheel']).lower(),

        # Profiling
        str(config['profiling']['nsys_on']).lower()
    ]

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
        print(f"\nProcessing: {config_file}")
        try:
            config = load_config(config_file)
            submit_job(config)
            print(f"Successfully submitted job for: {config_file}")
        except Exception as e:
            print(f"Error processing {config_file}: {e}", file=sys.stderr)
            # Continue processing other files even if one fails
            continue


if __name__ == '__main__':
    main()
