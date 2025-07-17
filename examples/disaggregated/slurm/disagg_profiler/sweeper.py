import argparse
import copy
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from functools import partial
from multiprocessing import Pool

import yaml

from .job_manager import JobManager, calculate_nodes_needed


def get_slurm_allocation(account, partition, time_limit, job_name, num_nodes):
    """
    Request a SLURM allocation and wait for it to be granted.

    Args:
        account (str): SLURM account
        partition (str): SLURM partition
        time_limit (str): Time limit for the allocation
        job_name (str): Job name
        num_nodes (int): Number of nodes to request

    Returns:
        str: Job ID of the SLURM allocation, or None if allocation failed
    """
    print(
        f"Requesting SLURM allocation for {num_nodes} nodes on partition {partition}..."
    )

    # capture SLURM_NODELIST inside allocation
    cmd = [
        "salloc", "-A", account, "-p", partition, "-N",
        str(num_nodes), "-t", time_limit, "-J", job_name, "--no-shell"
    ]

    try:
        # Run the command and stream the output in real-time
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   bufsize=1,
                                   universal_newlines=True)

        output_lines = []
        job_id = None

        # Process the output in real-time
        for line in process.stdout:
            print(line.strip())
            output_lines.append(line)

            # Look for job ID in the output (e.g., "salloc: Granted job allocation 12345")
            job_id_match = re.search(r"Granted job allocation (\d+)", line)
            if job_id_match:
                job_id = job_id_match.group(1)
                print(
                    f"Successfully obtained SLURM allocation with job ID: {job_id}"
                )
                break

        # Wait for the process to complete
        process.wait()

        if job_id:
            return job_id
        else:
            print(
                "Failed to obtain SLURM allocation. Check the output above for details."
            )
            return None
    except Exception as e:
        print(f"Error requesting SLURM allocation: {e}")
        return None


def run_sweep_configuration(config_params, args_dict):
    """
    Run a single sweep configuration in a separate process.

    Args:
        config_params (tuple): (config, description, config_num, total_configs, timestamp_base)
        args_dict (dict): Dictionary of arguments needed to run the configuration

    Returns:
        bool: True if successful, False otherwise
    """
    config, description, config_num, total_configs, timestamp_base = config_params

    print(f"Running configuration {config_num}/{total_configs}: {description}")

    # Reconstruct args from dict
    args = argparse.Namespace()
    for key, value in args_dict.items():
        setattr(args, key, value)

    # Calculate nodes needed for this specific configuration
    nodes_needed = calculate_nodes_needed(config, args.num_gpus)
    print(f"Configuration requires {nodes_needed} nodes")

    # Set a unique job ID with timestamp, index, and sweep parameters for better tracking
    job_id = f"sweep_{timestamp_base}_{config_num}of{total_configs}_{description}"

    # If allocation requested and not using a single allocation, allocate resources for just this configuration
    slurm_job_id = None
    if args.request_allocation and not args.use_single_allocation:
        print(
            f"Requesting allocation for configuration {config_num}/{total_configs}"
        )
        slurm_job_id = get_slurm_allocation(
            args.account, args.partition, args.time,
            f"{args.job_name}_sweep{config_num}", nodes_needed)

        if not slurm_job_id:
            print(
                f"Failed to get allocation for configuration {config_num}/{total_configs}, skipping"
            )
            return False

        # Set SLURM_JOB_ID for this run
        os.environ["SLURM_JOB_ID"] = slurm_job_id

    try:
        # Create and run the job manager with this configuration
        # JobManager will automatically check for existing results if --skip-existing is set
        with JobManager(args,
                        load_config_only=False,
                        override_config=config,
                        override_job_id=job_id) as job_manager:
            job_manager.launch_jobs()
        return True
    except Exception as e:
        print(f"Error running configuration {config_num}: {e}")
        return False
    finally:
        # Release the allocation if we requested one for this specific configuration
        if slurm_job_id:
            print(
                f"Releasing SLURM allocation (job ID: {slurm_job_id}) for configuration {config_num}"
            )
            subprocess.run(["scancel", slurm_job_id])
            print(f"SLURM allocation released for configuration {config_num}")


class AutoSweeper:
    """
    Automatically determines optimal sweep configurations by profiling context
    and generation servers separately to balance their request processing rates.

    This class:
    1. Profiles context-only servers
    2. Profiles generation-only servers
    3. Analyzes the performance ratios
    4. Generates sweep configurations to balance the rates
    """

    def __init__(self, args, max_gpus=None):
        """
        Initialize the auto sweeper.

        Args:
            args (argparse.Namespace): Command line arguments
            max_gpus (int): Maximum number of GPUs to use across all servers
        """
        self.args = args
        self.max_gpus = max_gpus or 32  # Default limit

        # Load base configuration
        temp_job_manager = JobManager(args, load_config_only=True)
        self.base_config = temp_job_manager.base_config

        # Storage for profiling results
        self.context_results = {}
        self.generation_results = {}

    def create_context_only_config(self, base_config, dp=1, tp=1):
        """
        Create a context-only configuration from a disaggregated config.

        Args:
            base_config (dict): Base disaggregated configuration
            dp (int): Data parallelism for context server
            tp (int): Tensor parallelism for context server

        Returns:
            dict: Context-only configuration
        """
        config = copy.deepcopy(base_config)

        # Remove generation and ifb servers
        if 'generation' in config['exec']['config']:
            del config['exec']['config']['generation']
        if 'ifb' in config['exec']['config']:
            del config['exec']['config']['ifb']

        # Configure context server - get base params from existing config or defaults
        base_context = config['exec']['config'].get('context', {})

        config['exec']['config']['ifb'] = {
            'tp': tp,
            'dp': dp,
            'ep': base_context.get('ep', 1),
            'pp': base_context.get('pp', 1),
            'max_batch_size': base_context.get('max_batch_size', 512),
            'max_seq_len': base_context.get('max_seq_len', 8192),
            'max_num_tokens': base_context.get('max_num_tokens', 8192),
            'config': base_context.get('config')
        }
        if 'context' in config['exec']['config']:
            del config['exec']['config']['context']

        # Set OSL to 1 for context profiling (we only care about context processing, not generation)
        if 'profile' in config:
            config['profile']['osl'] = 1
            print(
                f"Set OSL=1 for context profiling (was {config['profile'].get('osl', 'unknown')})"
            )

        return config

    def create_generation_only_config(self, base_config, dp=1, tp=1):
        """
        Create a generation-only configuration from a disaggregated config.

        Args:
            base_config (dict): Base disaggregated configuration
            dp (int): Data parallelism for generation server
            tp (int): Tensor parallelism for generation server

        Returns:
            dict: Generation-only configuration
        """
        config = copy.deepcopy(base_config)

        # Remove context servers
        if 'context' in config['exec']['config']:
            del config['exec']['config']['context']

        # Get base params from existing generation config or use ifb config as template
        base_gen = config['exec']['config'].get('generation', {})
        base_ifb = config['exec']['config'].get('ifb', {})

        # Use generation config as base, fallback to ifb, then defaults
        template_config = base_gen if base_gen else base_ifb

        # Configure as IFB (generation-only) server
        config['exec']['config']['generation'] = {
            'tp': tp,
            'dp': dp,
            'ep': template_config.get('ep', 1),
            'pp': template_config.get('pp', 1),
            'max_batch_size': template_config.get('max_batch_size', 512),
            'max_seq_len': template_config.get('max_seq_len', 8192),
            'max_num_tokens': template_config.get('max_num_tokens', 8192),
            'gen_only': True,  # Set generation-only mode
            'config': template_config.get('config')
        }

        # Remove the generation config since we're using IFB for gen-only
        if 'ifb' in config['exec']['config']:
            del config['exec']['config']['ifb']

        return config

    def parse_loadgen_results(self, output_folder):
        """
        Parse loadgen results to extract Requests/s for each concurrency level.

        Args:
            output_folder (str): Path to output folder containing results

        Returns:
            dict: Mapping of concurrency -> requests_per_second
        """
        results = {}

        # Look for concurrency subdirectories
        if not os.path.exists(output_folder):
            print(f"Output folder {output_folder} does not exist")
            return results

        for item in os.listdir(output_folder):
            concurrency_path = os.path.join(output_folder, item)
            if os.path.isdir(concurrency_path) and item.isdigit():
                concurrency = int(item)

                benchmark_serving_path = os.path.join(concurrency_path,
                                                      "benchmark_serving.log")
                if os.path.exists(benchmark_serving_path):
                    with open(benchmark_serving_path, 'r') as f:
                        for line in f:
                            if "Request throughput (req/s):" in line:
                                rps = float(
                                    line.split("Request throughput (req/s):")
                                    [1].strip())
                                results[concurrency] = rps
                                print(
                                    f"Concurrency {concurrency}: {rps:.3f} Requests/s"
                                )
                else:
                    # Look for raw_data.json file
                    raw_data_path = os.path.join(concurrency_path,
                                                 "raw_data.json")
                    if os.path.exists(raw_data_path):
                        try:
                            with open(raw_data_path, 'r') as f:
                                data = json.load(f)

                            if 'results' in data and 'infbench_summary' in data[
                                    'results']:
                                rps = data['results']['infbench_summary'][
                                    'Requests/s']
                                results[concurrency] = rps
                                print(
                                    f"Concurrency {concurrency}: {rps:.3f} Requests/s"
                                )
                        except Exception as e:
                            print(f"Error parsing {raw_data_path}: {e}")

        return results

    def profile_context_servers(self, tp_values=None, dp_values=None):
        """
        Profile context-only servers with different TP/DP configurations.

        Args:
            tp_values (list): List of TP values to test (if None, uses default from config)
            dp_values (list): List of DP values to test (if None, uses default from config)
        """
        if tp_values is None:
            # Extract default TP value from base configuration
            default_tp = 1
            if 'context' in self.base_config['exec']['config']:
                default_tp = self.base_config['exec']['config']['context'].get(
                    'tp', 1)
            elif 'ifb' in self.base_config['exec']['config']:
                default_tp = self.base_config['exec']['config']['ifb'].get(
                    'tp', 1)
            tp_values = [default_tp]
            print(f"Using default context TP value: {default_tp}")
        if dp_values is None:
            # Extract default DP value from base configuration
            default_dp = 1
            if 'context' in self.base_config['exec']['config']:
                default_dp = self.base_config['exec']['config']['context'].get(
                    'dp', 1)
            elif 'ifb' in self.base_config['exec']['config']:
                default_dp = self.base_config['exec']['config']['ifb'].get(
                    'dp', 1)
            dp_values = [default_dp]
            print(f"Using default context DP value: {default_dp}")

        print("=== Profiling Context Servers ===")
        print(f"DEBUG: tp_values = {tp_values}, dp_values = {dp_values}")
        print(
            f"DEBUG: Will run {len(tp_values)} x {len(dp_values)} = {len(tp_values) * len(dp_values)} iterations"
        )

        iteration_count = 0
        for tp in tp_values:
            for dp in dp_values:
                iteration_count += 1
                print(
                    f"DEBUG: Starting iteration {iteration_count} with TP={tp}, DP={dp}"
                )

                gpus_needed = tp * dp
                if gpus_needed > self.max_gpus:
                    print(
                        f"Skipping context TP={tp}, DP={dp} (needs {gpus_needed} GPUs > {self.max_gpus})"
                    )
                    continue

                print(f"Profiling context server: TP={tp}, DP={dp}")

                # Create context-only config
                config = self.create_context_only_config(self.base_config,
                                                         dp=dp,
                                                         tp=tp)

                # Calculate nodes needed for this configuration
                nodes_needed = calculate_nodes_needed(config,
                                                      self.args.num_gpus)

                # Request allocation for this specific configuration
                slurm_job_id = None
                if hasattr(
                        self.args,
                        'request_allocation') and self.args.request_allocation:
                    print(
                        f"Requesting allocation for context TP={tp}, DP={dp} ({nodes_needed} nodes)"
                    )
                    slurm_job_id = get_slurm_allocation(
                        self.args.account, self.args.partition, self.args.time,
                        f"{self.args.job_name}_ctx_tp{tp}_dp{dp}", nodes_needed)

                    if not slurm_job_id:
                        print(
                            f"Failed to get allocation for context TP={tp}, DP={dp}, skipping"
                        )
                        self.context_results[(tp, dp)] = {}
                        continue

                    # Set SLURM_JOB_ID for this run
                    os.environ["SLURM_JOB_ID"] = slurm_job_id
                    print(
                        f"Using allocation {slurm_job_id} for context TP={tp}, DP={dp}"
                    )

                # Set job ID for this configuration
                job_id = f"ctx_profile_tp{tp}_dp{dp}"

                try:
                    # Run the profiling
                    # JobManager will automatically check for existing results if --skip-existing is set
                    with JobManager(self.args,
                                    load_config_only=False,
                                    override_config=config,
                                    override_job_id=job_id) as job_manager:
                        job_manager.launch_jobs()

                    # Parse results
                    results = self.parse_loadgen_results(
                        job_manager.output_folder)
                    self.context_results[(tp, dp)] = results

                    print(f"Context TP={tp}, DP={dp} results: {results}")

                except Exception as e:
                    print(f"Error profiling context TP={tp}, DP={dp}: {e}")
                    self.context_results[(tp, dp)] = {}
                finally:
                    # Release the allocation if we requested one
                    if slurm_job_id:
                        print(
                            f"Releasing allocation {slurm_job_id} for context TP={tp}, DP={dp}"
                        )
                        subprocess.run(["scancel", slurm_job_id])
                        # Clear the environment variable
                        if "SLURM_JOB_ID" in os.environ:
                            del os.environ["SLURM_JOB_ID"]

        print(
            f"DEBUG: Completed context profiling with {iteration_count} total iterations"
        )

    def profile_generation_servers(self, tp_values=None, dp_values=None):
        """
        Profile generation-only servers with different TP/DP configurations.

        Args:
            tp_values (list): List of TP values to test (if None, uses default from config)
            dp_values (list): List of DP values to test (if None, uses default from config)
        """
        if tp_values is None:
            # Extract default TP value from base configuration
            default_tp = 1
            if 'generation' in self.base_config['exec']['config']:
                default_tp = self.base_config['exec']['config'][
                    'generation'].get('tp', 1)
            elif 'ifb' in self.base_config['exec']['config']:
                default_tp = self.base_config['exec']['config']['ifb'].get(
                    'tp', 1)
            tp_values = [default_tp]
            print(f"Using default generation TP value: {default_tp}")
        if dp_values is None:
            # Extract default DP value from base configuration
            default_dp = 1
            if 'generation' in self.base_config['exec']['config']:
                default_dp = self.base_config['exec']['config'][
                    'generation'].get('dp', 1)
            elif 'ifb' in self.base_config['exec']['config']:
                default_dp = self.base_config['exec']['config']['ifb'].get(
                    'dp', 1)
            dp_values = [default_dp]
            print(f"Using default generation DP value: {default_dp}")

        print("=== Profiling Generation Servers ===")
        print(f"DEBUG: tp_values = {tp_values}, dp_values = {dp_values}")
        print(
            f"DEBUG: Will run {len(tp_values)} x {len(dp_values)} = {len(tp_values) * len(dp_values)} iterations"
        )

        iteration_count = 0
        for tp in tp_values:
            for dp in dp_values:
                iteration_count += 1
                print(
                    f"DEBUG: Starting iteration {iteration_count} with TP={tp}, DP={dp}"
                )

                gpus_needed = tp * dp
                if gpus_needed > self.max_gpus:
                    print(
                        f"Skipping generation TP={tp}, DP={dp} (needs {gpus_needed} GPUs > {self.max_gpus})"
                    )
                    continue

                print(f"Profiling generation server: TP={tp}, DP={dp}")

                # Create generation-only config
                config = self.create_generation_only_config(self.base_config,
                                                            dp=dp,
                                                            tp=tp)

                # Calculate nodes needed for this configuration
                nodes_needed = calculate_nodes_needed(config,
                                                      self.args.num_gpus)

                # Request allocation for this specific configuration
                slurm_job_id = None
                if hasattr(
                        self.args,
                        'request_allocation') and self.args.request_allocation:
                    print(
                        f"Requesting allocation for generation TP={tp}, DP={dp} ({nodes_needed} nodes)"
                    )
                    slurm_job_id = get_slurm_allocation(
                        self.args.account, self.args.partition, self.args.time,
                        f"{self.args.job_name}_gen_tp{tp}_dp{dp}", nodes_needed)

                    if not slurm_job_id:
                        print(
                            f"Failed to get allocation for generation TP={tp}, DP={dp}, skipping"
                        )
                        self.generation_results[(tp, dp)] = {}
                        continue

                    # Set SLURM_JOB_ID for this run
                    os.environ["SLURM_JOB_ID"] = slurm_job_id
                    print(
                        f"Using allocation {slurm_job_id} for generation TP={tp}, DP={dp}"
                    )

                # Set job ID for this configuration
                job_id = f"gen_profile_tp{tp}_dp{dp}"

                try:
                    # Run the profiling
                    # JobManager will automatically check for existing results if --skip-existing is set
                    with JobManager(self.args,
                                    load_config_only=False,
                                    override_config=config,
                                    override_job_id=job_id) as job_manager:
                        job_manager.launch_jobs()

                    # Parse results
                    results = self.parse_loadgen_results(
                        job_manager.output_folder)
                    self.generation_results[(tp, dp)] = results

                    print(f"Generation TP={tp}, DP={dp} results: {results}")

                except Exception as e:
                    print(f"Error profiling generation TP={tp}, DP={dp}: {e}")
                    self.generation_results[(tp, dp)] = {}
                finally:
                    # Release the allocation if we requested one
                    if slurm_job_id:
                        print(
                            f"Releasing allocation {slurm_job_id} for generation TP={tp}, DP={dp}"
                        )
                        subprocess.run(["scancel", slurm_job_id])
                        # Clear the environment variable
                        if "SLURM_JOB_ID" in os.environ:
                            del os.environ["SLURM_JOB_ID"]

        print(
            f"DEBUG: Completed generation profiling with {iteration_count} total iterations"
        )

    def calculate_optimal_ratios(self):
        """
        Calculate optimal server ratios based on profiling results.

        Returns:
            list: List of optimal configurations with ratios (deduplicated)
        """
        print("=== Calculating Optimal Ratios ===")

        all_configs = []

        # For each concurrency level, find the best combinations
        all_concurrencies = set()

        # Collect all concurrency levels from both results
        for results in self.context_results.values():
            all_concurrencies.update(results.keys())
        for results in self.generation_results.values():
            all_concurrencies.update(results.keys())

        for concurrency in sorted(all_concurrencies):
            print(f"\nAnalyzing concurrency level: {concurrency}")

            # Compare each context config with each generation config
            for (ctx_tp, ctx_dp), ctx_results in self.context_results.items():
                if concurrency not in ctx_results:
                    continue

                ctx_rps = ctx_results[concurrency]

                for (gen_tp,
                     gen_dp), gen_results in self.generation_results.items():
                    if concurrency not in gen_results:
                        continue

                    gen_rps = gen_results[concurrency]

                    if gen_rps <= 0:
                        continue

                    # Calculate the ratio of context to generation throughput
                    ratio = ctx_rps / gen_rps

                    # Determine optimal server counts
                    if ratio < 1:
                        # Context is slower (bottleneck) - need more context servers
                        ctx_servers_needed = math.ceil(1 / ratio)
                        gen_servers_needed = 1
                    else:
                        # Generation is slower (bottleneck) - need more generation servers
                        gen_servers_needed = math.ceil(ratio)
                        ctx_servers_needed = 1

                    total_gpus = (ctx_servers_needed * ctx_tp * ctx_dp +
                                  gen_servers_needed * gen_tp * gen_dp)

                    if total_gpus <= self.max_gpus:
                        config = {
                            'concurrency':
                            concurrency,
                            'ctx_tp':
                            ctx_tp,
                            'ctx_dp':
                            ctx_dp,
                            'ctx_servers':
                            ctx_servers_needed,
                            'ctx_rps':
                            ctx_rps,
                            'gen_tp':
                            gen_tp,
                            'gen_dp':
                            gen_dp,
                            'gen_servers':
                            gen_servers_needed,
                            'gen_rps':
                            gen_rps,
                            'ratio':
                            ratio,
                            'total_gpus':
                            total_gpus,
                            'balanced_rps':
                            min(ctx_rps * ctx_servers_needed,
                                gen_rps * gen_servers_needed)
                        }
                        all_configs.append(config)

        # Deduplicate configurations based on actual server configuration
        # Key: (ctx_tp, ctx_dp, ctx_servers, gen_tp, gen_dp, gen_servers)
        unique_configs = {}

        for config in all_configs:
            # Create a key based on the actual configuration parameters that matter
            config_key = (config['ctx_tp'], config['ctx_dp'],
                          config['ctx_servers'], config['gen_tp'],
                          config['gen_dp'], config['gen_servers'])

            # Keep the configuration with the highest balanced RPS
            if config_key not in unique_configs or config[
                    'balanced_rps'] > unique_configs[config_key]['balanced_rps']:
                unique_configs[config_key] = config

        # Convert back to list and sort by balanced throughput
        optimal_configs = list(unique_configs.values())
        optimal_configs.sort(key=lambda x: x['balanced_rps'], reverse=True)

        print(
            f"\nFound {len(all_configs)} total configurations across all concurrency levels"
        )
        print(
            f"After deduplication: {len(optimal_configs)} unique configurations"
        )

        for config in optimal_configs:
            print(
                f"  Config: Ctx({config['ctx_tp']},{config['ctx_dp']})x{config['ctx_servers']} "
                f"+ Gen({config['gen_tp']},{config['gen_dp']})x{config['gen_servers']} "
                f"= {config['balanced_rps']:.2f} RPS using {config['total_gpus']} GPUs "
                f"(from concurrency {config['concurrency']})")

        return optimal_configs

    def generate_sweep_configurations(self, optimal_configs):
        """
        Generate sweep configurations based on optimal ratios.

        Args:
            optimal_configs (list): List of optimal configurations

        Returns:
            list: List of sweep configurations
        """
        print("=== Generating Sweep Configurations ===")

        sweep_configs = []

        for config in optimal_configs:
            # Create the disaggregated configuration
            disagg_config = copy.deepcopy(self.base_config)

            # Get base configs for proper parameter inheritance
            base_context = self.base_config['exec']['config'].get('context', {})
            base_generation = self.base_config['exec']['config'].get(
                'generation', {})

            # Configure context servers
            disagg_config['exec']['config']['context'] = {
                'tp': config['ctx_tp'],
                'dp': config['ctx_servers'],  # Use server count as DP
                'ep': base_context.get('ep', 1),
                'pp': base_context.get('pp', 1),
                'max_batch_size': base_context.get('max_batch_size', 512),
                'max_seq_len': base_context.get('max_seq_len', 8192),
                'max_num_tokens': base_context.get('max_num_tokens', 8192),
                'config': base_context.get('config', {})
            }

            # Configure generation servers
            disagg_config['exec']['config']['generation'] = {
                'tp': config['gen_tp'],
                'dp': config['gen_servers'],  # Use server count as DP
                'ep': base_generation.get('ep', 1),
                'pp': base_generation.get('pp', 1),
                'max_batch_size': base_generation.get('max_batch_size', 512),
                'max_seq_len': base_generation.get('max_seq_len', 8192),
                'max_num_tokens': base_generation.get('max_num_tokens', 8192),
                'gen_only': False,
                'config': base_generation.get('config', {})
            }

            # Remove ifb config if it exists (we're using disaggregated mode)
            if 'ifb' in disagg_config['exec']['config']:
                del disagg_config['exec']['config']['ifb']

            description = (
                f"c{config['concurrency']}_ctx{config['ctx_tp']}x{config['ctx_servers']}_"
                f"gen{config['gen_tp']}x{config['gen_servers']}")

            sweep_configs.append((disagg_config, description))

        print(f"Generated {len(sweep_configs)} sweep configurations")
        return sweep_configs

    def run_auto_sweep(self,
                       ctx_tp_values=None,
                       ctx_dp_values=None,
                       gen_tp_values=None,
                       gen_dp_values=None):
        """
        Run the complete auto-sweep process.

        Args:
            ctx_tp_values (list): TP values to test for context servers (if None, uses default from config)
            ctx_dp_values (list): DP values to test for context servers (if None, uses default from config)
            gen_tp_values (list): TP values to test for generation servers (if None, uses default from config)
            gen_dp_values (list): DP values to test for generation servers (if None, uses default from config)
        """
        print(f"Starting auto-sweep with max {self.max_gpus} GPUs")
        print(
            f"DEBUG: Called with ctx_tp_values={ctx_tp_values}, ctx_dp_values={ctx_dp_values}"
        )
        print(
            f"DEBUG: Called with gen_tp_values={gen_tp_values}, gen_dp_values={gen_dp_values}"
        )

        # Step 1: Profile context servers
        print("DEBUG: Step 1 - About to call profile_context_servers")
        self.profile_context_servers(ctx_tp_values, ctx_dp_values)
        print("DEBUG: Step 1 - Completed profile_context_servers")

        # Step 2: Profile generation servers
        print("DEBUG: Step 2 - About to call profile_generation_servers")
        self.profile_generation_servers(gen_tp_values, gen_dp_values)
        print("DEBUG: Step 2 - Completed profile_generation_servers")

        # Step 3: Calculate optimal ratios
        print("DEBUG: Step 3 - About to calculate optimal ratios")
        optimal_configs = self.calculate_optimal_ratios()

        if not optimal_configs:
            print("No optimal configurations found!")
            return

        # Step 4: Generate sweep configurations
        print("DEBUG: Step 4 - About to generate sweep configurations")
        sweep_configs = self.generate_sweep_configurations(optimal_configs)

        # Step 5: Run the sweep configurations
        if sweep_configs:
            print("=== Running Optimal Sweep Configurations ===")

            # Create a parameter sweeper to run the generated configurations
            # Ensure we use separate allocations for each configuration
            temp_args = copy.deepcopy(self.args)
            temp_args.use_single_allocation = False  # Force separate allocations

            sweeper = ParameterSweeper(
                "", temp_args)  # Empty config since we provide directly
            sweeper.run_generated_sweeps(sweep_configs)
        else:
            print("No sweep configurations generated!")


class ParameterSweeper:
    """
    Manages parameter sweeps for the disaggregated serving setup.

    This class handles the parsing of sweep configurations, generates all combinations
    of parameter values to be tested, and runs the corresponding jobs using JobManager.
    It abstracts away the details of parameter manipulation and provides a clean interface
    for running multiple configurations.

    Example usage:
        sweeper = ParameterSweeper('sweep_config.yaml', args)
        sweeper.run_sweeps()
    """

    def __init__(self, sweep_config_path, args):
        """
        Initialize the parameter sweeper.

        Args:
            sweep_config_path (str): Path to the sweep configuration file or a string
                                    representation of the sweep configuration.
            args (argparse.Namespace): Command line arguments.
        """
        self.sweep_config_path = sweep_config_path
        self.args = args
        self.job_manager = None
        self.sweep_params = self.parse_sweep_config()

    def parse_sweep_config(self):
        """
        Parse the sweep configuration string into a list of parameter dictionaries.

        The sweep configuration can be either:
        - A path to a YAML file containing the sweep definition
        - A string representation of a Python list of dictionaries
        - Empty string (for generated configs)

        Returns:
            List of dictionaries, where each dictionary defines a sweep.
        """
        # If no sweep config provided (for generated configs), return empty list
        if not self.sweep_config_path:
            return []

        # Load the sweep configuration from the file or string
        if os.path.exists(self.sweep_config_path):
            with open(self.sweep_config_path, 'r') as f:
                sweep_params = yaml.safe_load(f)
        else:
            # Try to parse as a Python dictionary
            try:
                sweep_params = eval(self.sweep_config_path)
            except Exception as e:
                print(
                    f"Error parsing sweep configuration: {self.sweep_config_path}"
                )
                print(f"Exception: {e}")
                sys.exit(1)

        return sweep_params

    def run_sweeps(self):
        """
        Generate all sweep configurations and run jobs for each one.

        This is the main method that orchestrates the sweeping process:
        1. Load the base configuration
        2. Generate all configurations from sweep parameters
        3. Run a job for each configuration with appropriate allocation
        """
        print(
            f"Running parameter sweep with configuration: {self.sweep_config_path}"
        )

        # Create a temporary job manager to load the base config
        temp_args = copy.deepcopy(self.args)
        temp_args.sweep_config = None  # Disable sweep for the temporary job manager

        temp_job_manager = JobManager(temp_args, load_config_only=True)
        base_config = temp_job_manager.base_config

        # Generate all configurations
        all_configs = self.generate_sweep_configurations(base_config)

        # Run the configurations
        self._run_sweep_configurations(all_configs)

    def run_generated_sweeps(self, sweep_configs):
        """
        Run sweep configurations that are already generated.

        Args:
            sweep_configs (list): List of (config, description) tuples
        """
        print(f"Running {len(sweep_configs)} generated sweep configurations")

        # Convert to the expected format
        all_configs = sweep_configs

        # Run the configurations
        self._run_sweep_configurations(all_configs)

    def _run_sweep_configurations(self, all_configs):
        """
        Internal method to run a list of sweep configurations.

        Args:
            all_configs (list): List of (config, description) tuples
        """
        # Check if any sweep configuration needs more nodes than the base configuration
        base_config = all_configs[0][0] if all_configs else {}
        base_nodes = calculate_nodes_needed(base_config)
        max_nodes = base_nodes

        for config, _ in all_configs:
            config_nodes = calculate_nodes_needed(config)
            if config_nodes > max_nodes:
                max_nodes = config_nodes

        if max_nodes > base_nodes:
            print(
                f"Warning: Some sweep configurations require more nodes ({max_nodes}) than the base configuration ({base_nodes})."
            )

        total_configs = len(all_configs)
        timestamp_base = time.strftime("%Y%m%d-%H%M%S")

        # If using a single allocation for all sweeps, get it now
        single_allocation_job_id = None
        if self.args.request_allocation and self.args.use_single_allocation:
            print(
                f"Requesting a single allocation for all sweep configurations")
            print(f"Using maximum node count: {max_nodes}")
            single_allocation_job_id = get_slurm_allocation(
                self.args.account, self.args.partition, self.args.time,
                f"{self.args.job_name}_sweep_all", max_nodes)

            if not single_allocation_job_id:
                print(
                    "Failed to get allocation for all configurations, exiting")
                return

            # Set SLURM_JOB_ID for this run
            os.environ["SLURM_JOB_ID"] = single_allocation_job_id
            print(
                f"Using single allocation with job ID: {single_allocation_job_id} for all configurations"
            )

        # Determine number of parallel processes to use
        num_parallel = min(self.args.parallel_sweeps, total_configs)
        print(
            f"Running {total_configs} configurations with {num_parallel} parallel processes"
        )

        # Prepare config parameters for parallel execution
        config_params = []
        for idx, (config, description) in enumerate(all_configs):
            config_num = idx + 1
            config_params.append((config, description, config_num,
                                  total_configs, timestamp_base))

        # Convert args to dictionary for pickling
        args_dict = vars(copy.deepcopy(self.args))
        args_dict[
            'sweep_config'] = None  # Disable sweep for the individual jobs

        try:
            # Run configurations in parallel if requested
            if num_parallel > 1:
                print(
                    f"Starting parallel execution with {num_parallel} processes"
                )
                with Pool(processes=num_parallel) as pool:
                    results = pool.map(
                        partial(run_sweep_configuration, args_dict=args_dict),
                        config_params)

                # Report results
                successful = sum(1 for r in results if r)
                print(
                    f"Completed {successful}/{total_configs} configurations successfully"
                )
            else:
                # Run configurations sequentially
                print("Starting sequential execution")
                for params in config_params:
                    run_sweep_configuration(params, args_dict)
        finally:
            # Release the single allocation if we requested one
            if single_allocation_job_id:
                print(
                    f"Releasing single SLURM allocation (job ID: {single_allocation_job_id})"
                )
                subprocess.run(["scancel", single_allocation_job_id])
                print("SLURM allocation released.")

    def generate_sweep_configurations(self, base_config):
        """
        Generate all configurations based on the sweep parameters.

        Args:
            base_config (dict): Base configuration to apply parameter changes to.

        Returns:
            List of (config, description) tuples, where config is the modified configuration
            and description is a string describing the parameter values.
        """
        all_configs = []

        # Process each sweep
        for sweep_idx, sweep in enumerate(self.sweep_params):
            param_names = []
            param_values = []

            for param_path, values in sweep.items():
                param_names.append(param_path)
                param_values.append(values)

            print(
                f"Generating configurations for sweep {sweep_idx+1}/{len(self.sweep_params)}"
            )

            # Generate all combinations for this sweep
            for idx, combination in enumerate(itertools.product(*param_values)):
                # Create a new config for this combination
                config = copy.deepcopy(base_config)

                # Apply the parameter values to the config
                param_description = []
                for i, param_path in enumerate(param_names):
                    value = combination[i]
                    self.set_config_value(config, param_path, value)
                    param_description.append(
                        f"{param_path.split('.')[-1]}={value}")

                combination_desc = "_".join(param_description)
                all_configs.append((config, combination_desc))

        return all_configs

    def set_config_value(self, config, param_path, value):
        """
        Set a configuration value based on a dot-notation path.

        Supports synchronized parameters using the "sync" prefix:
        - sync.tp: Sets both context.tp and generation.tp to the same value
        - sync.dp: Sets both context.dp and generation.dp to the same value
        - sync.max_batch_size: Sets both context.max_batch_size and generation.max_batch_size

        Args:
            config (dict): Configuration to modify
            param_path (str): Parameter path in dot notation
            value: Value to set
        """
        parts = param_path.split('.')

        # Handle synchronized parameters (sync prefix)
        if parts[0] == 'sync' and len(parts) > 1:
            sync_param = '.'.join(
                parts[1:])  # Join all parts after 'sync' to handle nested paths

            # Apply to both context and generation servers
            context_path = f"context.{sync_param}"
            generation_path = f"generation.{sync_param}"

            print(
                f"Synchronizing parameter {sync_param}={value} across context and generation servers"
            )

            # Set context server parameter
            if 'context' in config.get('exec', {}).get('config', {}):
                self.set_config_value(config, context_path, value)

            # Set generation server parameter
            if 'generation' in config.get('exec', {}).get('config', {}):
                self.set_config_value(config, generation_path, value)

            return

        # Special parameter name mappings for commonly used shortened names
        if parts[0] == 'context' and len(parts) > 1:
            parts = ['exec', 'config', 'context'] + parts[1:]
        elif parts[0] == 'generation' and len(parts) > 1:
            parts = ['exec', 'config', 'generation'] + parts[1:]
        elif parts[0] == 'gen' and len(parts) > 1:
            parts = ['exec', 'config', 'generation'] + parts[1:]
        elif parts[0] == 'ifb' and len(parts) > 1:
            parts = ['exec', 'config', 'ifb'] + parts[1:]

        # Navigate to the correct position in the config
        current = config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the value
        current[parts[-1]] = value
        print(f"Set {param_path} to {value}")


class MultiConfigSweeper:
    """
    Handles multiple configurations for context and generation servers, testing all combinations.

    This class supports configurations where context and generation servers can each have
    multiple predefined configurations, and all combinations of these configurations are tested.

    The configuration format allows either:
    1. Single configuration (existing format)
    2. List of configurations for exhaustive combination testing

    Example multi-config format:
    exec:
      config:
        context:
          - tp: 2
            dp: 1
            max_batch_size: 128
            config: {...}
          - tp: 4
            dp: 1
            max_batch_size: 256
            config: {...}
        generation:
          - tp: 2
            dp: 2
            max_batch_size: 64
            config: {...}
          - tp: 4
            dp: 1
            max_batch_size: 128
            config: {...}
    """

    def __init__(self, args):
        """
        Initialize the multi-config sweeper.

        Args:
            args (argparse.Namespace): Command line arguments
        """
        self.args = args

        # Load base configuration
        temp_job_manager = JobManager(args, load_config_only=True)
        self.base_config = temp_job_manager.base_config

    def is_multi_config(self, config):
        """
        Check if the configuration uses multiple configurations format.

        Args:
            config (dict): Configuration to check

        Returns:
            bool: True if multi-config format is detected
        """
        exec_config = config.get('exec', {}).get('config', {})

        # Check if context or generation sections are lists
        for section in ['context', 'generation', 'ifb']:
            if section in exec_config and isinstance(exec_config[section],
                                                     list):
                return True

        return False

    def extract_multi_configs(self, config):
        """
        Extract multiple configurations from the config.

        Args:
            config (dict): Configuration with potential multi-config format

        Returns:
            tuple: (context_configs, generation_configs, ifb_configs) where each is a list
        """
        exec_config = config.get('exec', {}).get('config', {})

        # Extract context configurations
        context_configs = []
        if 'context' in exec_config:
            if isinstance(exec_config['context'], list):
                context_configs = exec_config['context']
            else:
                context_configs = [exec_config['context']]

        # Extract generation configurations
        generation_configs = []
        if 'generation' in exec_config:
            if isinstance(exec_config['generation'], list):
                generation_configs = exec_config['generation']
            else:
                generation_configs = [exec_config['generation']]

        # Extract ifb configurations
        ifb_configs = []
        if 'ifb' in exec_config:
            if isinstance(exec_config['ifb'], list):
                ifb_configs = exec_config['ifb']
            else:
                ifb_configs = [exec_config['ifb']]

        return context_configs, generation_configs, ifb_configs

    def generate_config_combinations(self, base_config):
        """
        Generate all combinations of configurations.

        Args:
            base_config (dict): Base configuration

        Returns:
            list: List of (config, description) tuples for all combinations
        """
        if not self.is_multi_config(base_config):
            print(
                "Configuration is not in multi-config format, returning single config"
            )
            return [(base_config, "single_config")]

        context_configs, generation_configs, ifb_configs = self.extract_multi_configs(
            base_config)

        combinations = []
        combination_id = 0

        # Generate combinations based on available server types
        if context_configs and generation_configs:
            # Disaggregated mode: context + generation combinations
            print(
                f"Generating combinations for disaggregated mode: {len(context_configs)} context × {len(generation_configs)} generation"
            )

            for ctx_idx, ctx_config in enumerate(context_configs):
                for gen_idx, gen_config in enumerate(generation_configs):
                    combination_id += 1

                    # Create combined configuration
                    combined_config = copy.deepcopy(base_config)
                    combined_config['exec']['config']['context'] = ctx_config
                    combined_config['exec']['config']['generation'] = gen_config

                    # Remove ifb if present (disaggregated mode)
                    if 'ifb' in combined_config['exec']['config']:
                        del combined_config['exec']['config']['ifb']

                    # Generate description
                    description = f"ctx{ctx_idx+1}_gen{gen_idx+1}"

                    combinations.append((combined_config, description))

        elif ifb_configs:
            # IFB mode: single server combinations
            print(
                f"Generating combinations for IFB mode: {len(ifb_configs)} configurations"
            )

            for ifb_idx, ifb_config in enumerate(ifb_configs):
                combination_id += 1

                # Create combined configuration
                combined_config = copy.deepcopy(base_config)
                combined_config['exec']['config']['ifb'] = ifb_config

                # Remove context and generation if present (IFB mode)
                for section in ['context', 'generation']:
                    if section in combined_config['exec']['config']:
                        del combined_config['exec']['config'][section]

                # Generate description
                description = f"ifb{ifb_idx+1}"

                combinations.append((combined_config, description))

        else:
            print("Warning: No valid server configurations found")
            return [(base_config, "fallback_config")]

        print(f"Generated {len(combinations)} configuration combinations")
        return combinations

    def run_multi_config_sweep(self):
        """
        Run sweep with all configuration combinations.
        """
        print("=== Multi-Configuration Sweep Mode ===")
        print(
            "Generating all combinations of context and generation server configurations"
        )

        # Generate all configuration combinations
        all_configs = self.generate_config_combinations(self.base_config)

        if not all_configs:
            print("No configurations generated, exiting")
            return

        print(f"Will run {len(all_configs)} configuration combinations:")
        for i, (config, description) in enumerate(all_configs, 1):
            print(f"  {i}: {description}")

        # Use existing sweep infrastructure to run all combinations
        parameter_sweeper = ParameterSweeper(
            "", self.args)  # Empty sweep config path
        parameter_sweeper.run_generated_sweeps(all_configs)
