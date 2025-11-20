import argparse
import os
import socket
import time

import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ctx_servers",
                        type=int,
                        required=True,
                        help="Number of context servers")
    parser.add_argument("--num_gen_servers",
                        type=int,
                        required=True,
                        help="Number of generation servers")
    parser.add_argument("--work_dir",
                        type=str,
                        default="logs",
                        help="Work directory")
    parser.add_argument("--worker_port",
                        type=int,
                        default=8336,
                        help="Worker port")
    parser.add_argument("--server_port",
                        type=int,
                        default=8333,
                        help="Server port")
    args = parser.parse_args()

    # check if the work_dir exists
    if not os.path.exists(args.work_dir):
        raise ValueError(f"Work directory {args.work_dir} not found")

    #check all of the hostnames in the hostnames folder exists, if not, sleep 10 seconds and check again
    hostnames_folder = os.path.join(args.work_dir, "hostnames")
    while not os.path.exists(hostnames_folder):
        time.sleep(10)
        print(f"Waiting for hostnames folder {hostnames_folder} to be found")
    hostnames = os.listdir(hostnames_folder)
    # check length of hostnames is equal to num_ctx_servers + num_gen_servers, if not, sleep 10 seconds and check again
    while len(hostnames) != args.num_ctx_servers + args.num_gen_servers:
        time.sleep(10)
        hostnames = os.listdir(hostnames_folder)
        print(
            f"Waiting for hostnames to be found in {hostnames_folder}, current length: {len(hostnames)}, expected length: {args.num_ctx_servers + args.num_gen_servers}"
        )
    print(f"All hostnames found in {hostnames_folder}")

    # get the ctx and gen hostnames from the hostnames file
    ctx_hostnames = []
    gen_hostnames = []
    for hostname_file in hostnames:
        hostname_file_path = os.path.join(hostnames_folder, hostname_file)
        with open(hostname_file_path, 'r') as f:
            actual_hostname = f.read().strip()
            print(f"Hostname: {actual_hostname} in {hostname_file}")

        if hostname_file.startswith("CTX"):
            ctx_hostnames.append(actual_hostname)
        elif hostname_file.startswith("GEN"):
            gen_hostnames.append(actual_hostname)

    print(f"ctx_hostnames: {ctx_hostnames}")
    print(f"gen_hostnames: {gen_hostnames}")

    # get current hostname from env
    hostname = socket.gethostname()
    print(f"Current hostname: {hostname}")

    server_config = {
        'hostname': hostname,
        'port': args.server_port,
        'backend': 'pytorch',
        'context_servers': {
            'num_instances': args.num_ctx_servers,
            'urls': [f'{host}:{args.worker_port}' for host in ctx_hostnames]
        },
        'generation_servers': {
            'num_instances': args.num_gen_servers,
            'urls': [f'{host}:{args.worker_port}' for host in gen_hostnames]
        }
    }

    with open(os.path.join(args.work_dir, "server_config.yaml"), "w") as f:
        yaml.dump(server_config, f)
    print(
        f"Server config file {os.path.join(args.work_dir, 'server_config.yaml')} generated"
    )
