import argparse
import os
import subprocess
import math
from collections import defaultdict
from typing import List, Dict, Any
import json

def allocate_gpus(
    hostnames: List[str],
    base_port: int,
    gpus_per_node: int,
    num_gen_servers: int,
    num_ctx_servers: int,
    gen_world_size: int,
    ctx_world_size: int
) -> List[Dict[str, Any]]:

    allocations = []

    global_gpu_cursor = 0

    def get_gpu_location(gpus_per_node: int):
        node_id = global_gpu_cursor // gpus_per_node
        local_gpu_id = global_gpu_cursor % gpus_per_node
        return node_id, local_gpu_id

    def assign_server(server_allocation: Dict[str, Any], world_size: int, gpus_per_node: int):
        nonlocal global_gpu_cursor
        for _ in range(world_size):
            node_id, gpu_id = get_gpu_location(gpus_per_node)
            hostname = hostnames[node_id]
            if hostname not in server_allocation["nodes"]:
                server_allocation["nodes"][hostname] = []
            server_allocation["nodes"][hostname].append(gpu_id)
            global_gpu_cursor += 1

    def assign_servers(server_allocations: List[Dict[str, Any]], server_type: str, num_servers: int, world_size: int, gpus_per_node: int):
        for i in range(num_servers):
            server_allocation = {
                "server_type": server_type,
                "server_id": i,
                "port": base_port + i,
                "nodes": {}
            }
            assign_server(server_allocation, world_size, gpus_per_node)
            server_allocations.append(server_allocation)

    assign_servers(allocations, "GEN", num_gen_servers, gen_world_size, gpus_per_node)
    assign_servers(allocations, "CTX", num_ctx_servers, ctx_world_size, gpus_per_node)

    return allocations



def main():
    parser = argparse.ArgumentParser(description="Assign nodes and ports to servers")
    # parser.add_argument("--slurm-nodelist", type=str, required=True, help="SLURM_NODELIST")
    parser.add_argument("--all-nodes", type=str, required=True, help="All nodes")
    parser.add_argument("--gpus-per-node", type=int, required=True, help="GPUs per node")
    parser.add_argument("--num-gen-servers", type=int, required=True, help="Number of generation servers")
    parser.add_argument("--num-ctx-servers", type=int, required=True, help="Number of context servers")
    parser.add_argument("--gen-world-size", type=int, required=True, help="World size for generation servers")
    parser.add_argument("--ctx-world-size", type=int, required=True, help="World size for context servers")
    parser.add_argument("--base-port", type=int, default=8000, help="Base port number")
    parser.add_argument("--log-dir", type=str, required=True, help="Log directory")

    args = parser.parse_args()
    print(f"args: {args}")

    hostnames = args.all_nodes.split(',')
    total_nodes = len(hostnames)
    print(f"Total nodes: {total_nodes}")
    print(f"Hostnames: {hostnames}")

    allocations = allocate_gpus(hostnames=hostnames,
                                base_port=args.base_port,
                                gpus_per_node=args.gpus_per_node,
                                num_gen_servers=args.num_gen_servers,
                                num_ctx_servers=args.num_ctx_servers,
                                gen_world_size=args.gen_world_size,
                                ctx_world_size=args.ctx_world_size)
    print(json.dumps(allocations, indent=2))
    with open(os.path.join(args.log_dir, "allocations.json"), "w") as f:
        json.dump(allocations, f, indent=2)

if __name__ == "__main__":
    main()
