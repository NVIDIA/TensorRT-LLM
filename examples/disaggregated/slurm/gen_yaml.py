import argparse
import os
import re
from typing import Dict, List

import yaml


def process_node_and_task() -> tuple[int, List[str], List[str]]:
    """
    Process SLURM node and task environment variables.

    Returns:
        tuple: (max_tasks_per_node, nodes, task_nodes)
    """
    slurm_job_nodelist = os.getenv('SLURM_JOB_NODELIST', '')
    print(f"SLURM_JOB_NODELIST: {slurm_job_nodelist}")
    if not slurm_job_nodelist:
        raise ValueError(f"Environment variable SLURM_JOB_NODELIST not found.")

    slurm_tasks_per_node = os.getenv('SLURM_TASKS_PER_NODE', '')
    print(f"SLURM_TASKS_PER_NODE: {slurm_tasks_per_node}")
    if not slurm_tasks_per_node:
        raise ValueError(
            f"Environment variable SLURM_TASKS_PER_NODE not found.")

    # Generate list of nodes
    if '[' in slurm_job_nodelist:
        # Handle nodelist with range format
        node_prefix = slurm_job_nodelist.split('[')[
            0]  # Extract everything before '['
        node_range = re.search(r'\[(.*?)\]', slurm_job_nodelist).group(1)
        nodes = []
        for part in node_range.split(','):
            if '-' in part:
                start, end = part.split('-')
                # Get the width of the number format from the first number
                width = len(start)
                # Convert to integers after getting the width
                start, end = int(start), int(end)
                # Format numbers with leading zeros
                nodes.extend([
                    f"{node_prefix}{str(i).zfill(width)}"
                    for i in range(start, end + 1)
                ])
            else:
                # Preserve the original format for single numbers
                nodes.append(f"{node_prefix}{part}")
    else:
        # Handle single node format
        nodes = [slurm_job_nodelist]
    print(f"Nodes: {nodes}")

    # Generate tasks per node
    tasks_per_node = []
    for part in slurm_tasks_per_node.split(','):
        if '(x' in part:
            count, repeat = map(int, re.findall(r'\d+', part))
            tasks_per_node.extend([count] * repeat)
        else:
            tasks_per_node.append(int(part))
    print(f"Tasks per node: {tasks_per_node}")

    if (len(tasks_per_node) != len(nodes)):
        raise ValueError(
            f"Number of nodes and tasks per node do not match. Number of nodes: {len(nodes)}, Number of tasks per node: {len(tasks_per_node)}"
        )

    max_tasks_per_node = max(tasks_per_node)
    task_nodes = []
    for node, tasks in zip(nodes, tasks_per_node):
        task_nodes.extend([node] * tasks)

    return max_tasks_per_node, nodes, task_nodes


def generate_urls(ctx_or_gen: str,
                  num_instances: int,
                  tensor_parallel_size: int,
                  pipeline_parallel_size: int,
                  max_tasks_per_node: int,
                  nodes: List[str],
                  task_nodes: List[str],
                  node_to_port: Dict[str, int],
                  task_nodes_offset: int = 0) -> tuple[List[str], int]:
    """
    Generate URLs for context or generation servers.

    Returns:
        tuple: (urls, updated_task_nodes_offset)
    """
    urls = []

    for instance in range(num_instances):
        tasks_needed = tensor_parallel_size * pipeline_parallel_size

        if (task_nodes_offset + tasks_needed) > len(task_nodes):
            print(f"{ctx_or_gen} urls so far: {urls}")
            raise ValueError(
                f"For {ctx_or_gen} instance {instance}, there are not enough tasks available. task_nodes_offset: {task_nodes_offset}, tasks_needed: {tasks_needed}, len(task_nodes): {len(task_nodes)}"
            )

        min_node = (tasks_needed + max_tasks_per_node - 1) // max_tasks_per_node
        instance_nodes = set(task_nodes[task_nodes_offset:task_nodes_offset +
                                        tasks_needed])
        if len(instance_nodes) > min_node:
            raise ValueError(
                f"Tasks for a instance {instance} of {ctx_or_gen} instances use more node than expected. Nodes used: {instance_nodes}, number of nodes expected: {min_node}, max_tasks_per_node: {max_tasks_per_node}"
            )

        node = task_nodes[task_nodes_offset]
        port = node_to_port[node]
        node_to_port[node] += 1
        task_nodes_offset += tasks_needed

        urls.append(f"{node}:{port}")

    print(f"{ctx_or_gen} urls: {urls}")
    return urls, task_nodes_offset


def gen_config_file(config_path: str,
                    model_path: str,
                    num_ctx_servers: int,
                    ctx_tp_size: int,
                    ctx_batch_size: int,
                    ctx_max_num_tokens: int,
                    ctx_max_seq_len: int,
                    ctx_free_gpu_memory_fraction: float,
                    ctx_enable_attention_dp: bool,
                    num_gen_servers: int,
                    gen_tp_size: int,
                    gen_batch_size: int,
                    gen_max_num_tokens: int,
                    gen_max_seq_len: int,
                    gen_enable_attention_dp: bool,
                    gen_gpu_memory_fraction: float,
                    eplb_num_slots: int,
                    mtp_size: int = 0,
                    worker_start_port: int = 8001,
                    server_port: int = 8000,
                    cache_transceiver_max_num_tokens: int = 4608) -> None:
    """
    Generate configuration YAML file for disaggregated inference.

    Args:
        config_path: Path to save the config file
        model_path: Path to the model
        num_ctx_servers: Number of context servers
        ctx_tp_size: Tensor parallel size for context servers
        ctx_batch_size: Batch size for context servers
        ctx_max_num_tokens: Max number of tokens for context servers
        ctx_max_seq_len: Max sequence length for context servers
        ctx_free_gpu_memory_fraction: Free GPU memory fraction for context servers
        ctx_enable_attention_dp: Enable attention DP for context servers
        num_gen_servers: Number of generation servers
        gen_tp_size: Tensor parallel size for generation servers
        gen_batch_size: Batch size for generation servers
        gen_max_num_tokens: Max number of tokens for generation servers
        gen_enable_attention_dp: Enable attention DP for generation servers
        gen_gpu_memory_fraction: GPU memory fraction for generation servers
        eplb_num_slots: Number of slots for eplb
        worker_start_port: Start port for workers
        server_port: Server port
    """
    gen_cuda_graph_batch_sizes = [
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 2048, gen_batch_size
    ]

    gen_moe_backend = "CUTLASS"
    if gen_tp_size >= 16 and gen_enable_attention_dp:
        gen_moe_backend = "WIDEEP"
    if not gen_enable_attention_dp:
        gen_moe_backend = "TRTLLM"

    config = {
        'model': model_path,
        'hostname': 'localhost',
        'port': server_port,
        'backend': 'pytorch',
        'context_servers': {
            'num_instances': num_ctx_servers,
            'max_batch_size': ctx_batch_size,
            'max_num_tokens': ctx_max_num_tokens,
            'max_seq_len': ctx_max_seq_len,
            'free_gpu_memory_fraction': ctx_free_gpu_memory_fraction,
            'tensor_parallel_size': ctx_tp_size,
            'moe_expert_parallel_size': ctx_tp_size,
            'enable_attention_dp': ctx_enable_attention_dp,
            'pipeline_parallel_size': 1,
            'print_iter_log': True,
            'disable_overlap_scheduler': True,
            'kv_cache_config': {
                'enable_block_reuse': False,
                'free_gpu_memory_fraction': ctx_free_gpu_memory_fraction,
                'dtype': 'fp8',
            },
            'cache_transceiver_config': {
                'max_tokens_in_buffer': cache_transceiver_max_num_tokens,
                'backend': 'default',
            },
        },
        'generation_servers': {
            'num_instances': num_gen_servers,
            'tensor_parallel_size': gen_tp_size,
            'moe_expert_parallel_size': gen_tp_size,
            'enable_attention_dp': gen_enable_attention_dp,
            'pipeline_parallel_size': 1,
            'max_batch_size': gen_batch_size,
            'max_num_tokens': gen_max_num_tokens,
            'max_seq_len': gen_max_seq_len,
            'free_gpu_memory_fraction': gen_gpu_memory_fraction,
            'cuda_graph_config': {
                'enable_padding': True,
                'batch_sizes': gen_cuda_graph_batch_sizes,
            },
            'print_iter_log': True,
            'kv_cache_config': {
                'enable_block_reuse': False,
                'free_gpu_memory_fraction': gen_gpu_memory_fraction,
                'dtype': 'fp8',
            },
            'moe_config': {
                'backend': gen_moe_backend,
            },
            'cache_transceiver_config': {
                'max_tokens_in_buffer': cache_transceiver_max_num_tokens,
                'backend': 'default',
            },
            'stream_interval': 20,
        }
    }

    # Process nodes and generate URLs
    max_tasks_per_node, nodes, task_nodes = process_node_and_task()
    node_ports = {node: worker_start_port for node in nodes}

    # Generate URLs for context and generation servers
    ctx_urls, task_nodes_offset = generate_urls("ctx", num_ctx_servers,
                                                ctx_tp_size, 1,
                                                max_tasks_per_node, nodes,
                                                task_nodes, node_ports)
    if num_ctx_servers > 0:
        config['context_servers']['urls'] = ctx_urls

    gen_urls, _ = generate_urls("gen", num_gen_servers, gen_tp_size, 1,
                                max_tasks_per_node, nodes, task_nodes,
                                node_ports, task_nodes_offset)
    config['generation_servers']['urls'] = gen_urls

    # set the hostname to the first node
    config['hostname'] = nodes[0]

    if gen_tp_size == 8 and not gen_enable_attention_dp:
        config['generation_servers']['allreduce_strategy'] = "MNNVL"

    if eplb_num_slots > 0:
        moe_load_balancer_file = os.path.join(os.path.dirname(config_path),
                                              "moe_load_balancer.yaml")
        moe_load_balancer_config = {
            'num_slots': eplb_num_slots,
            'layer_updates_per_iter': 1
        }
        with open(moe_load_balancer_file, "w") as f:
            yaml.dump(moe_load_balancer_config,
                      f,
                      default_flow_style=False,
                      sort_keys=False)
        config['generation_servers']['moe_config'][
            'load_balancer'] = moe_load_balancer_file

    if mtp_size > 0:
        config['context_servers']['speculative_config'] = {
            'decoding_type': 'MTP',
            'num_nextn_predict_layers': mtp_size
        }
        config['generation_servers']['speculative_config'] = {
            'decoding_type': 'MTP',
            'num_nextn_predict_layers': mtp_size
        }

    # Write config to file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# gen main and args
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/tmp/config.yaml")
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="Path to the model")
    parser.add_argument("--num_ctx_servers",
                        type=int,
                        required=True,
                        help="Number of context servers")
    parser.add_argument("--ctx_tp_size",
                        type=int,
                        required=True,
                        help="Tensor parallel size for context servers")
    parser.add_argument("--ctx_batch_size",
                        type=int,
                        required=True,
                        help="Batch size for context servers")
    parser.add_argument("--ctx_max_num_tokens",
                        type=int,
                        required=True,
                        help="Max number of tokens for context servers")
    parser.add_argument("--ctx_max_seq_len",
                        type=int,
                        required=True,
                        help="Max sequence length for context servers")
    parser.add_argument("--ctx_free_gpu_memory_fraction",
                        type=float,
                        required=True,
                        help="Free GPU memory fraction for context servers")
    parser.add_argument("--ctx_enable_attention_dp",
                        dest='ctx_enable_attention_dp',
                        action='store_true',
                        help="Enable attention DP for context servers")
    parser.add_argument("--num_gen_servers",
                        type=int,
                        required=True,
                        help="Number of generation servers")
    parser.add_argument("--gen_tp_size",
                        type=int,
                        required=True,
                        help="Tensor parallel size for generation servers")
    parser.add_argument("--gen_batch_size",
                        type=int,
                        required=True,
                        help="Batch size for generation servers")
    parser.add_argument("--gen_max_num_tokens",
                        type=int,
                        required=True,
                        help="Max number of tokens for generation servers")
    parser.add_argument("--gen_max_seq_len",
                        type=int,
                        required=True,
                        help="Max sequence length for generation servers")
    parser.add_argument("--gen_enable_attention_dp",
                        dest='gen_enable_attention_dp',
                        action='store_true',
                        help="Enable attention DP for generation servers")
    parser.add_argument("--gen_gpu_memory_fraction",
                        type=float,
                        required=True,
                        help="GPU memory fraction for generation servers")
    parser.add_argument("--eplb_num_slots",
                        type=int,
                        default=0,
                        help="Number of slots for eplb")
    parser.add_argument("--mtp_size",
                        type=int,
                        default=0,
                        help="Number of nextn layers for MTP")
    parser.add_argument("--worker_start_port",
                        type=int,
                        default=8336,
                        help="Start port for workers")
    parser.add_argument("--server_port",
                        type=int,
                        default=8333,
                        help="Server port")
    parser.add_argument("--cache_transceiver_max_num_tokens",
                        type=int,
                        default=4608,
                        help="Max number of tokens for cache transceiver")

    args = parser.parse_args()

    gen_config_file(args.config, args.model, args.num_ctx_servers,
                    args.ctx_tp_size, args.ctx_batch_size,
                    args.ctx_max_num_tokens, args.ctx_max_seq_len,
                    args.ctx_free_gpu_memory_fraction,
                    args.ctx_enable_attention_dp, args.num_gen_servers,
                    args.gen_tp_size, args.gen_batch_size,
                    args.gen_max_num_tokens, args.gen_max_seq_len,
                    args.gen_enable_attention_dp, args.gen_gpu_memory_fraction,
                    args.eplb_num_slots, args.mtp_size, args.worker_start_port,
                    args.server_port, args.cache_transceiver_max_num_tokens)
