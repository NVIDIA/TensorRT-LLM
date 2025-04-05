import argparse
import os
import re

import yaml

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Update YAML configuration with SLURM node information.')
parser.add_argument(
    '--nodelist_env_var',
    type=str,
    default='SLURM_JOB_NODELIST',
    help=
    'Name of the env var that provides the list of nodes as dev[7-8,11,13] for example'
)
parser.add_argument(
    '--tasks_per_node_env_var',
    type=str,
    default='SLURM_TASKS_PER_NODE',
    help=
    'Name of the env var that provides the tasks per node as 8(x3),2 for example'
)
parser.add_argument('--disagg_server_port',
                    type=int,
                    default=8000,
                    help='The port to use for disagg server')
parser.add_argument('--worker_start_port',
                    type=int,
                    default=8001,
                    help='The starting port to use for workers')
parser.add_argument('--input_yaml',
                    type=str,
                    default='config.yaml',
                    help='Path to the input YAML file')
parser.add_argument('--output_yaml',
                    type=str,
                    default='output_config.yaml',
                    help='Path to the output YAML file')
args = parser.parse_args()

# Parse SLURM_JOB_NODELIST and SLURM_TASKS_PER_NODE from environment variables
print("---")
slurm_job_nodelist = os.getenv(args.nodelist_env_var, '')
if not slurm_job_nodelist:
    raise ValueError(f"Environment variable {args.nodelist_env_var} not found.")
print(f"{args.nodelist_env_var}: {slurm_job_nodelist}")
slurm_tasks_per_node = os.getenv(args.tasks_per_node_env_var, '')
if not slurm_tasks_per_node:
    raise ValueError(
        f"Environment variable {args.tasks_per_node_env_var} not found.")
print(f"{args.tasks_per_node_env_var}: {slurm_tasks_per_node}")
print("---")

# Generate list of nodes
node_prefix = re.match(r'^[a-zA-Z]+', slurm_job_nodelist).group(0)
node_range = re.search(r'\[(.*?)\]', slurm_job_nodelist).group(1)
nodes = []
for part in node_range.split(','):
    if '-' in part:
        start, end = map(int, part.split('-'))
        nodes.extend([f"{node_prefix}{i}" for i in range(start, end + 1)])
    else:
        nodes.append(f"{node_prefix}{part}")
print(f"Nodes: {nodes}")

# Generate tasks per node
tasks_per_node = []
for part in slurm_tasks_per_node.split(','):
    if '(x' in part:
        count, repeat = map(int, re.findall(r'\d+', part))
        tasks_per_node.extend([count] * repeat)
    else:
        tasks_per_node.append(int(part))
print(f"Tasks_per_node: {tasks_per_node}")

if (len(tasks_per_node) != len(nodes)):
    raise ValueError(
        f"Number of nodes and tasks per node do not match. Number of nodes: {len(nodes)}, Number of tasks per node: {len(tasks_per_node)}"
    )

max_tasks_per_node = max(tasks_per_node)
task_nodes = []
for node, tasks in zip(nodes, tasks_per_node):
    task_nodes.extend([node] * tasks)

print(f"Task nodes: {task_nodes}")
print("---")


# Function to generate URLs
def generate_urls(ctx_or_gen,
                  num_instances,
                  tensor_parallel_size,
                  pipeline_parallel_size,
                  max_task_per_node,
                  nodes,
                  task_nodes,
                  node_to_port,
                  task_nodes_offset=0):
    urls = []

    for instance in range(num_instances):
        tasks_needed = tensor_parallel_size * pipeline_parallel_size

        if (task_nodes_offset + tasks_needed) > len(task_nodes):
            print(f"{ctx_or_gen} urls so far: {urls}")
            raise ValueError(
                f"For {ctx_or_gen} instance {instance}, there are not enough tasks available. task_nodes_offset: {task_nodes_offset}, tasks_needed: {tasks_needed}, len(task_nodes): {len(task_nodes)}"
            )

        # Minimum number of nodes needed for that instance
        min_node = (tasks_needed + max_tasks_per_node - 1) / max_tasks_per_node
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


# Load the YAML file
with open(args.input_yaml, 'r') as file:
    config = yaml.safe_load(file)

# Keep track of the port number for each node
node_ports = {}
for node in nodes:
    node_ports[node] = args.worker_start_port

# Generate URLs for context_servers and generation_servers
context_urls, task_node_offset = generate_urls(
    "ctx", config['context_servers']['num_instances'],
    config['context_servers']['tensor_parallel_size'],
    config['context_servers']['pipeline_parallel_size'], max_tasks_per_node,
    nodes, task_nodes, node_ports)

generation_urls, _ = generate_urls(
    "gen", config['generation_servers']['num_instances'],
    config['generation_servers']['tensor_parallel_size'],
    config['generation_servers']['pipeline_parallel_size'], max_tasks_per_node,
    nodes, task_nodes, node_ports, task_node_offset)

# Update the YAML configuration
config['hostname'] = nodes[0]
config['port'] = args.disagg_server_port
config['context_servers']['urls'] = context_urls
config['generation_servers']['urls'] = generation_urls

# Save the updated YAML file
with open(args.output_yaml, 'w') as file:
    yaml.safe_dump(config, file, sort_keys=False)

print("YAML file updated successfully.")
