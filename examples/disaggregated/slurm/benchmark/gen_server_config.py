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

    # helper to get port
    def get_port(role, index, default_port):
        port_file = os.path.join(args.work_dir, f"{role.lower()}_port_{index}.txt")
        if os.path.exists(port_file):
            with open(port_file, 'r') as f:
                return int(f.read().strip())
        return default_port

    # sort hostnames to ensure order matches indices if possible, though strict matching by ID is better
    # The filenames are ROLE_ID.txt

    ctx_urls = []
    gen_urls = []

    for hostname_file in sorted(hostnames):
        hostname_file_path = os.path.join(hostnames_folder, hostname_file)
        with open(hostname_file_path, 'r') as f:
            actual_hostname = f.read().strip()
            print(f"Hostname: {actual_hostname} in {hostname_file}")

        # Parse role and id
        # Expected format: ROLE_ID.txt
        filename_no_ext = os.path.splitext(hostname_file)[0]
        parts = filename_no_ext.split('_')
        if len(parts) >= 2:
            role_prefix = parts[0]
            instance_id = parts[1]

            port = get_port(role_prefix, instance_id, args.worker_port)
            url = f"{actual_hostname}:{port}"

            if role_prefix == "CTX":
                ctx_urls.append(url)
            elif role_prefix == "GEN":
                gen_urls.append(url)
        else:
            print(f"Skipping malformed filename: {hostname_file}")

    print(f"ctx_urls: {ctx_urls}")
    print(f"gen_urls: {gen_urls}")

    # get current hostname from env
    hostname = socket.gethostname()
    print(f"Current hostname: {hostname}")

    server_config = {
        'hostname': hostname,
        'port': args.server_port,
        'backend': 'pytorch',
        'context_servers': {
            'num_instances': args.num_ctx_servers,
            'urls': ctx_urls
        },
        'generation_servers': {
            'num_instances': args.num_gen_servers,
            'urls': gen_urls
        }
    }

    with open(os.path.join(args.work_dir, "server_config.yaml"), "w") as f:
        yaml.dump(server_config, f)
    print(
        f"Server config file {os.path.join(args.work_dir, 'server_config.yaml')} generated"
    )
