import json
import os
import subprocess
import time
import uuid

import requests
import yaml


def wait_for_server(host, port, max_retries=1000, delay=5):
    """Wait for a server to become available."""
    for _ in range(max_retries):
        try:
            print(f"Checking server status at {host}:{port}")
            state = requests.get(f"http://{host}:{port}/health")
            if state.status_code == 200:
                print("Server is running")
                return True
        except:
            time.sleep(delay)

    print(f"Server did not start after {max_retries} attempts")
    return False


def calculate_nodes_needed(config, num_gpus):
    """
    Calculate the number of nodes needed based on the configuration.

    The calculation is based on:
    1. The number of context servers and their TP size
    2. The number of generation servers and their TP size
    3. The number of IFB servers and their TP size
    4. The number of GPUs per node (num_gpus)

    Args:
        config (dict): The configuration dictionary

    Returns:
        int: The number of nodes needed
    """
    total_gpus_needed = 0

    # Calculate GPUs needed for context servers
    if 'context' in config['exec']['config']:
        context_config = config['exec']['config']['context']
        context_tp = context_config['tp']
        context_dp = context_config['dp']
        total_gpus_needed += context_tp * context_dp

    # Calculate GPUs needed for generation servers
    if 'generation' in config['exec']['config']:
        generation_config = config['exec']['config']['generation']
        generation_tp = generation_config['tp']
        generation_dp = generation_config['dp']
        total_gpus_needed += generation_tp * generation_dp

    # Calculate GPUs needed for IFB servers
    if 'ifb' in config['exec']['config']:
        ifb_config = config['exec']['config']['ifb']
        ifb_tp = ifb_config['tp']
        ifb_dp = ifb_config['dp']
        total_gpus_needed += ifb_tp * ifb_dp

    # Calculate nodes needed (round up to ensure enough nodes)
    nodes_needed = (total_gpus_needed + num_gpus - 1) // num_gpus

    # Ensure we request at least one node
    return max(1, nodes_needed)


class JobManager:
    """Manages SLURM jobs for the disaggregated serving setup."""

    def __init__(self,
                 args,
                 load_config_only=False,
                 override_config=None,
                 override_job_id=None):
        """
        Initialize the job manager.

        Args:
            args (argparse.Namespace): Command line arguments.
            load_config_only (bool): If True, only load the configuration without launching jobs.
            override_config (dict): Override the base config with this config if provided.
            override_job_id (str): Override the auto-generated job ID with this ID if provided.
        """
        print(
            f"DEBUG JobManager: __init__ called with load_config_only={load_config_only}, override_job_id={override_job_id}"
        )

        self.args = args
        self.num_gpus = args.num_gpus
        self.account = args.account
        self.partition = args.partition
        self.time = args.time
        self.job_name = args.job_name
        self.container_image = args.container_image
        self.mounts = args.mounts
        self.workdir = os.path.abspath(os.path.join(os.getcwd()))
        self.context_jobs = []
        self.generation_jobs = []
        self.ifb_jobs = []
        self.disagg_jobs = []
        # Track jobs that have nsys enabled with their nsys file paths
        self.nsys_jobs = []
        self.experiment_path = args.experiment_path

        # Clean up any previous node info files
        self.setup_shared_directories(override_job_id)

        # Generate configuration files
        self.base_config = self.load_config()

        # If override_config is provided, use it instead of the base_config
        if override_config:
            self.config = override_config
        else:
            self.config = self.base_config
        print(self.config)

        # If we only need to load the config, return here
        if load_config_only:
            print(
                "DEBUG JobManager: load_config_only=True, returning without launching jobs"
            )
            return

        self.output_folder = self.create_output_folder()

        # Check if results already exist and should be skipped
        if hasattr(args, 'skip_existing') and args.skip_existing:
            if self._check_existing_results():
                print(
                    f"Skipping execution - results already exist in {self.output_folder}"
                )
                return

        # Run the job with the current configuration
        print("DEBUG JobManager: About to call prepare_and_run_single_config")
        self.prepare_and_run_single_config()
        print("DEBUG JobManager: Completed prepare_and_run_single_config")

    def create_output_folder(self):
        # The parent directory must be model name
        model_name = self.config['exec']['model_path'].split('/')[-1]
        if model_name == '':
            model_name = self.config['exec']['model_path'].split('/')[-2]
        model_name = os.path.join(self.experiment_path, model_name)
        os.makedirs(model_name, exist_ok=True)
        output_folder = os.path.join(model_name, (
            f"{self.config['profile']['isl']}_{self.config['profile']['osl']}"))
        os.makedirs(output_folder, exist_ok=True)
        ifb_tp = self.config['exec']['config']['ifb'][
            'tp'] if 'ifb' in self.config['exec']['config'] else 0
        ifb_dp = self.config['exec']['config']['ifb'][
            'dp'] if 'ifb' in self.config['exec']['config'] else 0
        ctx_tp = self.config['exec']['config']['context'][
            'tp'] if 'context' in self.config['exec']['config'] else 0
        ctx_dp = self.config['exec']['config']['context'][
            'dp'] if 'context' in self.config['exec']['config'] else 0
        gen_tp = self.config['exec']['config']['generation'][
            'tp'] if 'generation' in self.config['exec']['config'] else 0
        gen_dp = self.config['exec']['config']['generation'][
            'dp'] if 'generation' in self.config['exec']['config'] else 0

        output_folder = os.path.join(output_folder, (f"{ctx_tp}_"
                                                     f"{ctx_dp}_"
                                                     f"{gen_tp}_"
                                                     f"{gen_dp}_"
                                                     f"{ifb_tp}_"
                                                     f"{ifb_dp}"))

        for server_type in ['generation', 'ifb']:
            if f'{server_type}' in self.config['exec']['config']:
                config = self.config['exec']['config'][f'{server_type}'][
                    'config']
                pytorch_config = config[
                    'pytorch_backend_config'] if 'pytorch_backend_config' in config else config
                if 'disable_overlap_scheduler' not in pytorch_config or not pytorch_config[
                        'disable_overlap_scheduler']:
                    output_folder += '_overlap'
                if 'use_cuda_graph' in pytorch_config:
                    if pytorch_config['use_cuda_graph']:
                        output_folder += '_cuda_graph'
                if 'enable_attention_dp' in config:
                    if config['enable_attention_dp']:
                        output_folder += '_adp'
                if 'speculative_config' in config:
                    if "num_nextn_predict_layers" in config[
                            "speculative_config"]:
                        output_folder += f'_mtp{config["speculative_config"]["num_nextn_predict_layers"]}'
                config = self.config['exec']['config'][f'{server_type}']
                if 'env' in config:
                    for key, value in config['env'].items():
                        output_folder += f'_{key}_{value}'

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        return output_folder

    def _check_existing_results(self):
        """
        Check if results already exist for this configuration.

        Returns:
            bool: True if all results exist and are valid, False otherwise
        """
        if not os.path.exists(self.output_folder):
            return False

        # Get concurrency levels from the configuration
        concurrency_levels = self.config.get('profile',
                                             {}).get('concurrency', [1])

        for concurrency in concurrency_levels:
            concurrency_path = os.path.join(self.output_folder,
                                            str(concurrency))
            raw_data_path = os.path.join(concurrency_path, "raw_data.json")

            if not os.path.exists(raw_data_path):
                return False

            # Check if the raw_data.json is valid and contains expected results
            try:
                with open(raw_data_path, 'r') as f:
                    data = json.load(f)

                if 'results' not in data or 'infbench_summary' not in data[
                        'results']:
                    return False

                if 'Requests/s' not in data['results']['infbench_summary']:
                    return False
            except (json.JSONDecodeError, KeyError):
                return False

        return True

    def setup_shared_directories(self, override_job_id=None):
        """Create shared directories for coordination between nodes."""

        # Use a unique directory for each job, or the override_job_id if provided
        if override_job_id:
            job_id = override_job_id
        else:
            job_id = str(uuid.uuid4())

        os.makedirs(f"job_{job_id}", exist_ok=True)
        self.job_id = job_id

    def load_config(self):
        """Load the configuration file."""
        with open(self.args.config_file, "r") as f:
            return yaml.safe_load(f)

    def create_context_config(self):
        """Create the context server configuration file."""
        if 'context' in self.config['exec']['config']:
            context_config = self.config['exec']['config']['context']['config']

            # Create config in temporary job directory for execution
            config_path = os.path.join(f"job_{self.job_id}", "context.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(context_config, f)

            # Also save a copy to the output folder for reference
            if hasattr(self, 'output_folder') and self.output_folder:
                output_config_path = os.path.join(self.output_folder,
                                                  "context_config.yaml")
                try:
                    with open(output_config_path, 'w') as f:
                        yaml.dump(context_config,
                                  f,
                                  default_flow_style=False,
                                  indent=2)
                    print(
                        f"Context configuration saved to {output_config_path}")
                except Exception as e:
                    print(
                        f"Warning: Could not save context config to output folder: {e}"
                    )

            return os.path.abspath(config_path)
        return None

    def create_generation_config(self):
        """Create the generation server configuration file."""
        if 'generation' in self.config['exec']['config']:
            generation_config = self.config['exec']['config']['generation'][
                'config']

            # Create config in temporary job directory for execution
            config_path = os.path.join(f"job_{self.job_id}", "generation.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(generation_config, f)

            # Also save a copy to the output folder for reference
            if hasattr(self, 'output_folder') and self.output_folder:
                output_config_path = os.path.join(self.output_folder,
                                                  "generation_config.yaml")
                try:
                    with open(output_config_path, 'w') as f:
                        yaml.dump(generation_config,
                                  f,
                                  default_flow_style=False,
                                  indent=2)
                    print(
                        f"Generation configuration saved to {output_config_path}"
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not save generation config to output folder: {e}"
                    )

            return os.path.abspath(config_path)
        return None

    def create_ifb_config(self):
        """Create the IFB server configuration file."""
        if 'ifb' in self.config['exec']['config']:
            ifb_config = self.config['exec']['config']['ifb']['config']

            # Create config in temporary job directory for execution
            config_path = os.path.join(f"job_{self.job_id}", "ifb.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(ifb_config, f)

            # Also save a copy to the output folder for reference
            if hasattr(self, 'output_folder') and self.output_folder:
                output_config_path = os.path.join(self.output_folder,
                                                  "ifb_config.yaml")
                try:
                    with open(output_config_path, 'w') as f:
                        yaml.dump(ifb_config,
                                  f,
                                  default_flow_style=False,
                                  indent=2)
                    print(f"IFB configuration saved to {output_config_path}")
                except Exception as e:
                    print(
                        f"Warning: Could not save IFB config to output folder: {e}"
                    )

            return os.path.abspath(config_path)
        return None

    def get_node_hostnames(self):
        """Get list of hostnames from SLURM allocation or fallback to localhost."""
        node_hostnames = []

        # Try to get hostnames from SLURM
        try:
            # Check if we're running in a SLURM allocation
            if "SLURM_JOB_ID" in os.environ:
                cmd = [
                    "srun", "--overlap", "--oversubscribe", "scontrol", "show",
                    "hostnames"
                ]
                print(f"Running command: {' '.join(cmd)}")
                output = subprocess.check_output(cmd).decode().strip().split(
                    '\n')
                for hostname in output:
                    if hostname.strip():
                        node_hostnames.append(hostname.strip())

                if not node_hostnames:
                    raise Exception("No hostnames returned by SLURM")
            else:
                raise Exception(
                    "Not running in a SLURM allocation (SLURM_JOB_ID not set)")
        except Exception as e:
            print(
                f"Error getting node hostnames: {e}, falling back to localhost")
            node_hostnames = ["localhost"]

        print(f"Using node hostnames: {node_hostnames}")
        return node_hostnames

    def calculate_server_distribution(self):
        """Calculate how to distribute context and generation servers across nodes."""
        distribution = []

        total_ctx_servers = self.config['exec']['config']['context'][
            'dp'] if 'context' in self.config['exec']['config'] else 0
        total_gen_servers = self.config['exec']['config']['generation'][
            'dp'] if 'generation' in self.config['exec']['config'] else 0
        total_ifb_servers = self.config['exec']['config']['ifb'][
            'dp'] if 'ifb' in self.config['exec']['config'] else 0
        port_base = 8001

        gpu_index = 0
        server_id = 0
        j = 0
        for ctx_server in range(total_ctx_servers):
            gpu_indices = []
            server_ids = []
            server_ids.append(self.node_hostnames[server_id])
            for i in range(self.config['exec']['config']['context']['tp']):
                if i != 0 and i % self.num_gpus == 0:
                    server_id += 1
                    print(f'Adding {self.node_hostnames[server_id]}')
                    server_ids.append(self.node_hostnames[server_id])
                gpu_indices.append(str(gpu_index % self.num_gpus))
                gpu_index += 1
                gpu_index = gpu_index % self.num_gpus
            if len(gpu_indices) > self.num_gpus:
                gpu_indices = gpu_indices[:self.num_gpus]

            distribution.append({
                "node_id": server_id,
                "hostnames": server_ids,
                "gpu_indices": gpu_indices,
                "is_generation": False,
                "is_ifb": False,
                "port": port_base + j
            })
            j += 1
            if gpu_index % self.num_gpus == 0:
                gpu_index = 0
                server_id += 1

        # If the last context server is not on a full node, add a generation server to the next node
        if gpu_index % self.num_gpus != 0:
            server_id += 1
            gpu_index = 0

        for gen_server in range(total_gen_servers):
            gpu_indices = []
            server_ids = []
            server_ids.append(self.node_hostnames[server_id])
            for i in range(self.config['exec']['config']['generation']['tp']):
                if i != 0 and i % self.num_gpus == 0:
                    server_id += 1
                    server_ids.append(self.node_hostnames[server_id])
                    print(f'Adding {self.node_hostnames[server_id]}')
                gpu_indices.append(str(gpu_index % self.num_gpus))
                gpu_index += 1
                gpu_index = gpu_index % self.num_gpus

            if len(gpu_indices) > self.num_gpus:
                gpu_indices = gpu_indices[:self.num_gpus]

            distribution.append({
                "node_id": server_id,
                "hostnames": server_ids,
                "gpu_indices": gpu_indices,
                "is_ifb": False,
                "is_generation": True,
                "port": port_base + j
            })
            j += 1
            if gpu_index % self.num_gpus == 0:
                gpu_index = 0
                server_id += 1
        for ifb_server in range(total_ifb_servers):
            gpu_indices = []
            server_ids = []
            server_ids.append(self.node_hostnames[server_id])
            for i in range(self.config['exec']['config']['ifb']['tp']):
                if i != 0 and i % self.num_gpus == 0:
                    server_id += 1
                    server_ids.append(self.node_hostnames[server_id])
                    print(f'Adding {self.node_hostnames[server_id]}')
                gpu_indices.append(str(gpu_index % self.num_gpus))
                gpu_index += 1
                gpu_index = gpu_index % self.num_gpus
            if len(gpu_indices) > self.num_gpus:
                gpu_indices = gpu_indices[:self.num_gpus]

            distribution.append({
                "node_id": server_id,
                "hostnames": server_ids,
                "gpu_indices": gpu_indices,
                "is_generation": False,
                "is_ifb": True,
                "port": port_base + j
            })
            j += 1
        return distribution

    def run_in_trtllm_container(self, cmd, num_tasks=1):
        """Run a command in the TRTLLM container."""
        cmd = [
            "srun", "--overlap", "--oversubscribe", "-A", self.account, "-p",
            self.partition, "-N", "1", "-n",
            str(num_tasks), "-t", self.time, "--container-image",
            self.container_image, "--container-mounts", self.mounts,
            "--container-workdir", self.workdir, "--mpi", "pmix", "bash", "-c",
            " ".join(cmd)
        ]
        return subprocess.run(cmd)

    def spawn_in_trtllm_container(self, cmd, hostname):
        """Spawn a command in the TRTLLM container."""
        cmd = [
            "srun", "--overlap", "--oversubscribe", "-A", self.account, "-p",
            self.partition, "-N", "1", "-t", self.time, "--container-image",
            self.container_image, "--container-mounts", self.mounts,
            "--container-workdir", self.workdir, "--mpi", "pmix", "-w",
            hostname, "bash", "-c", " ".join(cmd)
        ]
        return subprocess.Popen(cmd)

    def write_node_distribution(self, distribution):
        """Write the server distribution to a file for the disaggregated server."""
        # Ensure the job_id directory exists
        os.makedirs(f"job_{self.job_id}", exist_ok=True)

        with open(f"job_{self.job_id}/node_distribution.yaml", "w") as f:
            yaml.dump(distribution, f)

    def launch_jobs(self):
        """Launch SLURM jobs for all nodes."""
        print("DEBUG JobManager: launch_jobs started")

        # Check if results already exist and should be skipped
        if hasattr(self.args, 'skip_existing') and self.args.skip_existing:
            if self._check_existing_results():
                print(
                    f"Skipping job execution - results already exist in {self.output_folder}"
                )
                return

        # Calculate server distribution
        distribution = self.calculate_server_distribution()
        self.write_node_distribution(distribution)

        # Track all launched jobs
        # Launch context and generation servers on each node
        for node in distribution:
            node_id = node["node_id"]
            hostnames = node["hostnames"]
            gpu_indices = node["gpu_indices"]
            print(
                f"Launching context server on {hostnames[0]}:{node['port']}, GPUs: {gpu_indices}"
            )

            # Launch context servers
            if node["is_ifb"]:
                if 'use_trtllm_bench' not in self.config[
                        'profile'] or not self.config['profile'][
                            'use_trtllm_bench']:
                    log_file = f"ifb_server_{node_id}.log"
                    ifb_jobs = self.launch_ifb_server(node_id, hostnames,
                                                      gpu_indices, node["port"],
                                                      log_file)
                    self.ifb_jobs.extend(ifb_jobs)
            elif not node["is_generation"]:
                log_file = f"context_server_{node_id}.log"
                ctx_jobs = self.launch_context_server(node_id, hostnames,
                                                      gpu_indices, node["port"],
                                                      log_file)
                self.context_jobs.extend(ctx_jobs)
            else:
                log_file = f"generation_server_{node_id}.log"
                gen_jobs = self.launch_generation_server(
                    node_id, hostnames, gpu_indices, node["port"], log_file)
                self.generation_jobs.extend(gen_jobs)
            # Wait for server to start
            time.sleep(15)

        # Wait for all servers to start (give some time for them to initialize)
        print("Waiting for all servers to start...")

        for node in distribution:
            hostnames = node["hostnames"]
            print(f"Waiting for {hostnames[0]}:{node['port']} to start...")
            if 'use_trtllm_bench' not in self.config[
                    'profile'] or not self.config['profile']['use_trtllm_bench']:
                wait_for_server(hostnames[0], node["port"])

        # Collect server URLs and configure the disaggregated server
        context_server_urls = []
        generation_server_urls = []
        ifb_server_urls = []

        # Add all known server URLs based on the distribution
        for node in distribution:
            node_id = node["node_id"]
            hostnames = node["hostnames"]
            gpu_indices = node["gpu_indices"]

            # Add context server URLs
            node_port = node["port"]
            if node["is_ifb"]:
                ifb_server_urls.append(f"{hostnames[0]}:{node_port}")
            elif not node["is_generation"]:
                context_server_urls.append(f"{hostnames[0]}:{node_port}")
            else:
                generation_server_urls.append(f"{hostnames[0]}:{node_port}")

        # Launch disaggregated server on the master node
        distribution[0]["hostnames"][0]
        port = 8000

        if len(distribution) == 1 and distribution[0]["is_ifb"]:
            distribution[0]["hostnames"][0]
            port = distribution[0]["port"]
        else:
            for i in range(self.args.num_disagg_servers):
                disagg_job = self.launch_disaggregated_server(
                    self.node_hostnames[i], context_server_urls,
                    generation_server_urls)
                self.disagg_jobs.append(disagg_job)

            # Wait a bit for the disaggregated server to initialize
            print("Waiting for disaggregated server to initialize...")
            time.sleep(30)

        load_gen_jobs = []
        # Run the loadgen tests
        print("DEBUG JobManager: About to call run_loadgen_tests")
        for i in range(self.args.num_disagg_servers):
            load_gen_jobs.append(
                self.run_loadgen_tests(self.node_hostnames[i], port,
                                       generation_server_urls))
        print("DEBUG JobManager: Completed run_loadgen_tests")

        for loadgen_job in load_gen_jobs:
            loadgen_job.wait()

        
        self.terminate_jobs(self.context_jobs + self.generation_jobs +
                            self.ifb_jobs + self.disagg_jobs)
        print("DEBUG JobManager: launch_jobs completed")
    
    def stop_nsys_sessions(self):
        """Stop nsys sessions for all tracked nsys jobs."""
        if not self.nsys_jobs:
            return
        
        print("Stopping nsys sessions...")
        
        # Group jobs by hostname to minimize srun calls
        jobs_by_hostname = {}
        for nsys_job in self.nsys_jobs:
            if nsys_job['process'] and nsys_job['process'].poll() is None:
                # Use the first hostname for the job
                hostname = nsys_job['hostnames'][0]
                if hostname not in jobs_by_hostname:
                    jobs_by_hostname[hostname] = []
                jobs_by_hostname[hostname].append(nsys_job)
        
        # Process each hostname
        for hostname, jobs in jobs_by_hostname.items():
            try:
                print(f"Checking nsys sessions on node {hostname}")
                
                # First, get list of active sessions on this node
                list_cmd = [
                    "srun", "--overlap", "--oversubscribe", "-A", self.account, "-p", self.partition,
                    "-N", "1", "-n", "1", "-t", "00:05:00", "-w", hostname,
                    f"--container-image={self.container_image}",
                    f"--container-mounts={self.mounts}",
                    f"--container-workdir={self.workdir}",
                    "nsys", "sessions", "list", "--show-header=false"
                ]
                
                result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0:
                    print(f"Warning: Could not list nsys sessions on {hostname}")
                    print(f"Error output: {result.stderr}")
                    continue
                    
                active_sessions = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        # Sessions list format: typically session_id followed by other info
                        parts = line.strip().split()
                        if parts:
                            active_sessions.append(parts[0])  # First column is session ID
                
                print(f"Found {len(active_sessions)} active nsys sessions on {hostname}")
                
                # Stop sessions for jobs on this hostname
                for nsys_job in jobs:
                    session_name = nsys_job['session_name']
                    print(f"Stopping nsys session '{session_name}' for {nsys_job['server_type']} server (node {nsys_job['node_id']}) on {hostname}")
                    
                    # Try to stop by session name first
                    stop_cmd = [
                        "srun", "--overlap", "--oversubscribe", "-A", self.account, "-p", self.partition,
                        "-N", "1", "-n", "1", "-t", self.time, "-w", hostname,
                        f"--container-image={self.container_image}",
                        f"--container-mounts={self.mounts}",
                        f"--container-workdir={self.workdir}",
                        "nsys", "stop", f"--session={session_name}"
                    ]
                    
                    result = subprocess.run(stop_cmd, capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0:
                        print(f"Successfully stopped nsys session '{session_name}' on {hostname}")
                    else:
                        print(f"Warning: Could not stop session '{session_name}' on {hostname}. Error: {result.stderr}")
                    
                # Give nsys some time to write the profile data
                time.sleep(3)
                    
            except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
                print(f"Warning: Could not stop nsys sessions on {hostname}: {e}")
                    
        print("Finished stopping nsys sessions.")

    def _build_server_launch_command(self, config, hostnames, gpu_indices,
                                     node_port, config_path, log_file):
        """Build the command to launch a server (context or generation).
        
        Returns:
            tuple: (cmd, nsys_info) where nsys_info is a dict with nsys details or None
        """
        if not gpu_indices or not config_path:
            return None, None

        # Get configuration based on server type
        server_config = config
        tp = server_config['tp']
        ep = server_config['ep']
        pp = server_config['pp']
        max_batch_size = server_config['max_batch_size']
        max_seq_len = server_config['max_seq_len']
        max_num_tokens = server_config['max_num_tokens']
        cuda_visible_devices = ",".join(gpu_indices)

        print(
            f"Launching server on {hostnames[0]}:{node_port}, GPUs: {cuda_visible_devices}"
        )

        envs = ""
        envs += "TRTLLM_USE_UCX_KVCACHE=1 "
        envs += " CUDA_VISIBLE_DEVICES=" + cuda_visible_devices
        envs += " UCX_CUDA_IPC_ENABLE_MNNVL=n "
        if 'gen_only' in config and config['gen_only']:
            envs += " TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1"

        if 'env' in config:
            for key, value in config['env'].items():
                envs += f" {key}={value}"
        unset_env = ''
        if 'unset_env' in config:
            for key in config['unset_env']:
                unset_env += f" -u {key}"
        nsys = False
        nsys_prefix = ''
        nsys_info = None
        if 'nsys' in config:
            nsys = config['nsys']
        if nsys:
            envs += " TLLM_PROFILE_RECORD_GC=1"
            envs += " TLLM_NVTX_DEBUG=1"
            nsys_file = os.path.join(self.output_folder, f"{log_file}.nsys-rep")
            # Use a session name that includes the log file basename for easier identification
            session_name = f"trtllm_{os.path.basename(log_file)}"
            nsys_prefix = f"nsys profile -e \"NSYS_MPI_STORE_TEAMS_PER_RANK=1\" -o {nsys_file} -f true -t cuda,nvtx,python-gil -c cudaProfilerApi --cuda-graph-trace node --capture-range-end=stop --gpu-metrics-devices=none --session={session_name}"
            nsys_info = {
                'enabled': True,
                'nsys_file': nsys_file,
                'session_name': session_name,
                'hostnames': hostnames
            }
        log_file = os.path.join(self.output_folder, log_file)

        # Build the command
        cmd = [
            "srun",
            "--overlap",
            "--oversubscribe",
            "-A",
            self.account,
            "-p",
            self.partition,
            "-N",
            f"{len(hostnames)}",
            "-n",
            str(tp),
            "-t",
            self.time,
            "--mpi",
            "pmix",
            "-w",
            ",".join(hostnames),  # Specify the hostnames
            f"--container-image={self.container_image}",
            f"--container-mounts={self.mounts}",
            f"--container-workdir={self.workdir}",
            "--export=\"'CUDA_VISIBLE_DEVICES=" + cuda_visible_devices + "'\"",
            "bash",
            "-c",
            envs +
            " env " + unset_env + " " + nsys_prefix + " trtllm-llmapi-launch trtllm-serve " +
            self.config['exec']['model_path'] + " --host 0.0.0.0 --port " +
            str(node_port) + " --backend pytorch --extra_llm_api_options " +
            config_path + " --tp_size " + str(tp) + " --ep_size " + str(ep) +
            " --max_batch_size " + str(max_batch_size) + " --max_seq_len " +
            str(max_seq_len) + " --max_num_tokens " + str(max_num_tokens) +
            " --pp_size " + str(pp) + " &> " + log_file
        ]

        return cmd, nsys_info

    def launch_context_server(self, node_id, hostnames, gpu_indices, node_port,
                              log_file):
        """Launch context server on a specific node."""
        launched_jobs = []

        cmd, nsys_info = self._build_server_launch_command(
            self.config['exec']['config']['context'], hostnames, gpu_indices,
            node_port, self.context_config_path, log_file)
        if not cmd:
            return launched_jobs

        print(f"Running command: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd)
        launched_jobs.append(proc)

        # Track nsys information if enabled
        if nsys_info:
            nsys_info['process'] = proc
            nsys_info['server_type'] = 'context'
            nsys_info['node_id'] = node_id
            self.nsys_jobs.append(nsys_info)

        # Small delay to avoid race conditions in resource allocation
        time.sleep(2)

        return launched_jobs

    def launch_generation_server(self, node_id, hostnames, gpu_indices,
                                 node_port, log_file):
        """Launch generation server on a specific node."""
        launched_jobs = []

        cmd, nsys_info = self._build_server_launch_command(
            self.config['exec']['config']['generation'], hostnames, gpu_indices,
            node_port, self.generation_config_path, log_file)
        if not cmd:
            return launched_jobs

        print(f"Running command: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd)
        launched_jobs.append(proc)

        # Track nsys information if enabled
        if nsys_info:
            nsys_info['process'] = proc
            nsys_info['server_type'] = 'generation'
            nsys_info['node_id'] = node_id
            self.nsys_jobs.append(nsys_info)

        # Small delay to avoid race conditions in resource allocation
        time.sleep(2)

        return launched_jobs

    def launch_ifb_server(self, node_id, hostnames, gpu_indices, node_port,
                          log_file):
        """Launch IFB server on a specific node."""
        launched_jobs = []

        cmd, nsys_info = self._build_server_launch_command(
            self.config['exec']['config']['ifb'], hostnames, gpu_indices,
            node_port, self.ifb_config_path, log_file)
        if not cmd:
            return launched_jobs

        print(f"Running command: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd)
        launched_jobs.append(proc)

        # Track nsys information if enabled
        if nsys_info:
            nsys_info['process'] = proc
            nsys_info['server_type'] = 'ifb'
            nsys_info['node_id'] = node_id
            self.nsys_jobs.append(nsys_info)

        return launched_jobs

    def launch_disaggregated_server(self, hostname, context_servers,
                                    generation_servers):
        """Launch the disaggregated server on the master node."""
        if not generation_servers:
            print(
                "No context or generation servers available. Cannot launch disaggregated server."
            )
            return None

        config = self.config['exec']['config']
        # Create the disaggregated server configuration
        disagg_config = {
            "hostname": hostname,
            "port": 8000,
            "backend": "pytorch",
            "model": self.config['exec']['model_path'],
            "context_servers": {
                "num_instances":
                len(context_servers),
                "urls":
                context_servers,
                "tensor_parallel_size":
                config['context']['tp'] if 'context' in config else 1,
            },
            "generation_servers": {
                "num_instances":
                len(generation_servers),
                "urls":
                generation_servers,
                "tensor_parallel_size":
                self.config['exec']['config']['generation']['tp'],
            }
        }

        # Write the configuration to a file
        disagg_config_path = os.path.join(f"job_{self.job_id}",
                                          f"disagg_config_{hostname}.yaml")
        with open(disagg_config_path, 'w') as f:
            yaml.dump(disagg_config, f)
        disagg_config_path = os.path.abspath(disagg_config_path)

        # Also save a copy to the output folder for reference
        if hasattr(self, 'output_folder') and self.output_folder:
            output_config_path = os.path.join(self.output_folder,
                                              f"disagg_config_{hostname}.yaml")
            try:
                with open(output_config_path, 'w') as f:
                    yaml.dump(disagg_config,
                              f,
                              default_flow_style=False,
                              indent=2)
                print(
                    f"Disaggregated server configuration saved to {output_config_path}"
                )
            except Exception as e:
                print(
                    f"Warning: Could not save disaggregated config to output folder: {e}"
                )

        print(f"Launching disaggregated server on {hostname}:8000")
        envs = ""
        gen_only = self.config['exec']['config']['generation'].get(
            'gen_only', False)
        if gen_only:
            envs += " TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1"

        # Store the log to the output folder
        log_file = os.path.join(self.output_folder,
                                f"disagg_server_{hostname}.log")
        # Build the command
        cmd = [
            "srun",
            "--overlap",
            "--oversubscribe",
            "-A",
            self.account,
            "-p",
            self.partition,
            "-N",
            "1",
            "-n",
            "1",
            "-t",
            self.time,
            "-w",
            hostname,  # Specify the hostname
            "-J",
            f"{self.job_name}_disagg",
            "--mpi",
            "pmix",
            f"--container-image={self.container_image}",
            f"--container-mounts={self.mounts}",
            f"--container-workdir={self.workdir}",
            "--export=OMPI_ALLOW_RUN_AS_ROOT=1",
            "bash",
            "-c",
            f" {envs} trtllm-serve disaggregated -c " + disagg_config_path +
            " --request_timeout " + " 1800000" + " &> " + log_file
        ]

        print(f"Running command: {' '.join(cmd)}")
        return subprocess.Popen(cmd)

    def run_loadgen_tests(self, hostname, port, generation_server_urls):
        """Run loadgen tests on the master node."""
        print(
            f"DEBUG JobManager: run_loadgen_tests called with hostname={hostname}, port={port}"
        )
        output_folder = self.output_folder

        # Run loadgen tests for each concurrency level
        if 'use_trtllm_bench' in self.config['profile'] and self.config[
                'profile']['use_trtllm_bench']:
            self.prepare_dataset(output_folder)

        concurrency_levels = self.config['profile']['concurrency']
        print(
            f"DEBUG JobManager: Will run loadgen for concurrency levels: {concurrency_levels}"
        )

        for concurrency in concurrency_levels:
            print(
                f"DEBUG JobManager: Starting loadgen for concurrency {concurrency}"
            )
            concurrency_output_folder = f"{output_folder}/{concurrency}"
            if not os.path.exists(concurrency_output_folder):
                os.makedirs(concurrency_output_folder)

            # Build and run the loadgen command
            if 'use_trtllm_bench' in self.config['profile'] and self.config[
                    'profile']['use_trtllm_bench']:
                self.run_trtllm_bench_for_concurrency(
                    hostname, port, concurrency, concurrency_output_folder)
            elif 'use_benchmark_serving' in self.config[
                    'profile'] and self.config['profile'][
                        'use_benchmark_serving']:
                pid = self.run_benchmark_serving_for_concurrency(
                    hostname, port, concurrency, concurrency_output_folder)
                pid.wait()
            print(
                f"DEBUG JobManager: Completed loadgen for concurrency {concurrency}"
            )

        print("DEBUG JobManager: run_loadgen_tests completed")
        return pid

    def record_kv_cache_stats(self, generation_server_urls, output_folder):
        """Record the kv cache stats for a specific concurrency level."""
        print(
            f"DEBUG JobManager: Recording kv cache stats for generation servers"
        )
        # Get the kv cache stats
        # Send a request to "/metrics" and store the output in a file
        for url in generation_server_urls:
            response = requests.get(f"http://{url}/metrics")
            with open(
                    os.path.join(output_folder,
                                 f"{url.split(':')[0]}_kv_cache_stats.txt"),
                    "w") as f:
                f.write(response.text)
        print(
            f"DEBUG JobManager: Recorded kv cache stats for generation servers")

    def prepare_dataset(self, output_folder):
        """Prepare the dataset for the loadgen tests."""
        input_tokens = str(self.config['profile']['isl'])
        output_tokens = str(self.config['profile']['osl'])
        num_prompts = str(
            8 * self.config['exec']['config']['ifb']['max_batch_size'])

        # If the dataset file already exists, skip the preparation
        dataset_file = os.path.join(output_folder, "dataset.json")
        if os.path.exists(dataset_file):
            return

        cmd = [
            "python3", self.args.prepare_dataset_script, "--stdout",
            "--tokenizer", self.config['exec']['model_path'], "token-norm-dist",
            "--input-mean", input_tokens, "--output-mean", output_tokens,
            "--input-stdev", "0", "--output-stdev", "0", "--num-requests",
            num_prompts, ">", dataset_file
        ]
        print(f"Running command: {' '.join(cmd)}")
        self.run_in_trtllm_container(cmd)

    def run_trtllm_bench_for_concurrency(self, hostname, port, concurrency,
                                         output_folder):
        """Run the trtllm bench for a specific concurrency level."""
        config = self.config['exec']['config']['ifb']
        tp = config['tp'] if 'tp' in config else 0
        ep = config['ep'] if 'ep' in config else 0
        max_batch_size = config[
            'max_batch_size'] if 'max_batch_size' in config else 0
        max_num_tokens = config[
            'max_num_tokens'] if 'max_num_tokens' in config else 0
        gpu_fraction = config['config']['kv_cache_config'][
            'free_gpu_memory_fraction'] if 'config' in config and 'kv_cache_config' in config[
                'config'] else 0
        num_requests = concurrency * 8
        # Parent of output_folder is the dataset file
        dataset_file = os.path.join(os.path.dirname(output_folder),
                                    "dataset.json")
        # Redirect the output to a file
        cmd = [
            "trtllm-llmapi-launch", "trtllm-bench", "-m",
            "deepseek-ai/DeepSeek-R1", "--model_path",
            f"{self.config['exec']['model_path']}", "throughput", "--tp",
            f"{tp}", "--ep", f"{ep}", "--warmup", "0", "--dataset",
            f"{dataset_file}", "--backend", "pytorch", "--max_batch_size",
            f"{max_batch_size}", "--max_num_tokens", f"{max_num_tokens}",
            "--kv_cache_free_gpu_mem_fraction", f"{gpu_fraction}",
            "--extra_llm_api_options", f"{self.ifb_config_path}",
            "--num_requests", f"{num_requests}", "--concurrency",
            f"{concurrency}", "--report_json",
            f"{output_folder}/trtllm_bench_throughput.json", "--streaming",
            "&>" + os.path.join(output_folder, "output.log")
        ]
        print(f"Running command: {' '.join(cmd)}")
        self.run_in_trtllm_container(cmd, tp)

    def run_benchmark_serving_for_concurrency(self, hostname, port, concurrency,
                                              output_folder):
        """Run the benchmark serving test for a specific concurrency level."""
        model_name = self.config['exec']['model_path']
        dataset_path = self.config['profile']['dataset_path']
        num_prompts = self.config['profile']['num_prompts']
        cmd = [
            "/usr/bin/python3", "-m",
            "tensorrt_llm.serve.scripts.benchmark_serving", "--model",
            model_name, "--tokenizer", model_name, "--dataset-name",
            "trtllm_custom", "--dataset-path", dataset_path, "--num-prompts",
            str(num_prompts), "--host", hostname, "--port",
            str(port), "--max-concurrency",
            str(concurrency)
        ]
        if 'ignore_eos' in self.config['profile'] and self.config['profile'][
                'ignore_eos']:
            cmd += ["--ignore-eos"]
        cmd += ["&> " + os.path.join(output_folder, "benchmark_serving.log")]
        print(f"Running command: {' '.join(cmd)}")
        return self.spawn_in_trtllm_container(cmd, hostname)

    def terminate_jobs(self, jobs):
        """Terminate all running jobs."""
        # First, stop any nsys sessions for jobs that have nsys enabled
        self.stop_nsys_sessions()
        
        # Then terminate the jobs
        for job in jobs:
            if job and job.poll() is None:
                try:
                    job.terminate()
                    job.wait(timeout=5)
                except:
                    job.kill()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate_jobs(self.context_jobs + self.generation_jobs +
                            self.ifb_jobs + self.disagg_jobs)

    def save_config_to_output_folder(self):
        """Save the complete configuration used for this run to the output folder."""
        if not hasattr(self, 'output_folder') or not self.output_folder:
            return

        config_file_path = os.path.join(self.output_folder, "config.yaml")
        try:
            with open(config_file_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            print(f"Configuration saved to {config_file_path}")
        except Exception as e:
            print(
                f"Warning: Could not save configuration to output folder: {e}")

    def prepare_and_run_single_config(self):
        """Prepare and run a single configuration."""
        print("DEBUG JobManager: prepare_and_run_single_config started")

        # Create the configuration files
        self.context_config_path = self.create_context_config()
        self.generation_config_path = self.create_generation_config()
        self.ifb_config_path = self.create_ifb_config()

        # Save the complete configuration to the output folder
        self.save_config_to_output_folder()

        # Get hostnames of available nodes
        self.node_hostnames = self.get_node_hostnames()
