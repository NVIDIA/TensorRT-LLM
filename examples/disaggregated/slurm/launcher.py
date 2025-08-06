import argparse
import os
import subprocess
import sys

from disagg_profiler.job_manager import JobManager, calculate_nodes_needed
from disagg_profiler.sweeper import (AutoSweeper, MultiConfigSweeper,
                                     ParameterSweeper, get_slurm_allocation)


def setup_argparse():
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="TRT-LLM Disaggregated Serving Launcher")

    # SLURM job arguments
    parser.add_argument("--account",
                        type=str,
                        required=True,
                        help="SLURM account")
    parser.add_argument("--partition",
                        type=str,
                        required=True,
                        help="SLURM partition")
    parser.add_argument("--time",
                        type=str,
                        default="04:00:00",
                        help="SLURM time limit")
    parser.add_argument("--job-name",
                        type=str,
                        required=True,
                        help="SLURM job name")
    parser.add_argument("--container-image",
                        type=str,
                        required=True,
                        help="Container image")
    parser.add_argument("--mounts",
                        type=str,
                        required=True,
                        help="Container mount points")
    parser.add_argument("--num-gpus",
                        type=int,
                        required=True,
                        help="Number of GPUs on each node")
    parser.add_argument("--config-file",
                        type=str,
                        default="./config.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--experiment-path",
                        type=str,
                        default=".",
                        help="Path to the configuration file")
    parser.add_argument(
        "--prepare-dataset-script",
        type=str,
        default=None,
        help=
        "Path to the prepare dataset script. Required if trtllm_bench is used.")
    parser.add_argument(
        "--sweep-config",
        type=str,
        help=
        """Parameter sweep configuration as a Python string or a path to a YAML file.

                        Format:
                        - A list of dictionaries, where each dictionary defines a sweep
                        - Each dictionary maps parameter paths to lists of values to try

                        Example: '[{"context.dp": [1, 2, 3, 4]}, {"gen.dp": [1, 2, 3]}]'

                        This will first sweep context.dp with values 1-4 (keeping gen.dp=1),
                        then sweep gen.dp with values 1-3 (keeping context.dp=1).

                        Parameter paths can use shortened names:
                        - context.dp (for exec.config.context.dp)
                        - gen.dp (for exec.config.generation.dp)
                        - ifb.dp (for exec.config.ifb.dp)

                        Or full paths like exec.config.context.dp

                        For multiple parameters, use:
                        '[{"context.tp": [2, 4, 8], "context.dp": [1, 2, 4]}]'
                        This will try all combinations of tp and dp (9 combinations total).
                        """)
    parser.add_argument(
        "--auto-sweep",
        action="store_true",
        help=
        """Automatically determine optimal sweep configurations by profiling context and generation servers.

                        This mode will:
                        1. Profile context-only servers with different TP/DP configurations
                        2. Profile generation-only servers with different TP/DP configurations
                        3. Analyze the performance ratios to determine optimal server counts
                        4. Generate and run sweep configurations that balance context and generation throughput

                        Auto-sweep uses separate SLURM allocations for each configuration, providing better
                        resource utilization and fault tolerance compared to a single large allocation.

                        The goal is to ensure the request processing rates of context and generation servers are balanced.
                        If ctx_rps / gen_rps < 1, more generation servers are needed to match context throughput.
                        If ctx_rps / gen_rps > 1, more context servers are needed to match generation throughput.

                        Both the profiling phases and final sweep execution can be parallelized using --parallel-sweeps.
                        """)
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=32,
        help=
        "Maximum number of GPUs to use across all servers in auto-sweep mode (default: 32)"
    )
    parser.add_argument(
        "--request-allocation",
        action="store_true",
        help=
        "Request a SLURM allocation before running jobs. This is useful for interactive testing."
    )
    parser.add_argument(
        "--use-single-allocation",
        action="store_true",
        help=
        "When running sweeps, use a single allocation for all configurations instead of separate allocations for each."
    )
    parser.add_argument(
        "--parallel-sweeps",
        type=int,
        default=1,
        help="""Number of sweep configurations to run in parallel (default: 1).

                        This applies to:
                        - Manual sweep mode: final sweep configurations
                        - Auto-sweep mode: profiling phases AND final sweep configurations

                        For auto-sweep, this parallelizes both context server profiling, generation server profiling,
                        and the final optimal configuration sweep. This can significantly reduce total runtime.
                        """)
    parser.add_argument("--skip-existing",
                        action="store_true",
                        help="""Skip configurations that already have results.

                        This is useful for resuming interrupted sweeps or re-running experiments.
                        The launcher will check for existing output folders with results and skip
                        those configurations, only running new or incomplete ones.

                        This applies to:
                        - Manual sweep mode: skip sweep configurations with existing results
                        - Auto-sweep mode: skip profiling runs AND final sweep configurations with existing results
                        """)
    parser.add_argument("--num-disagg-servers",
                        type=int,
                        default=1,
                        help="Number of disaggregated servers to use.")

    return parser


def main():
    parser = setup_argparse()
    args = parser.parse_args()

    # Load base configuration to check for multi-config format
    temp_job_manager = JobManager(args, load_config_only=True)
    base_config = temp_job_manager.base_config

    # Auto-detect multi-config format
    multi_sweeper = MultiConfigSweeper(args)
    is_multi_config = multi_sweeper.is_multi_config(base_config)

    # Validate arguments - can't use multiple sweep modes at once
    sweep_modes = [args.auto_sweep, bool(args.sweep_config), is_multi_config]
    active_modes = []
    if args.auto_sweep:
        active_modes.append("--auto-sweep")
    if args.sweep_config:
        active_modes.append("--sweep-config")
    if is_multi_config:
        active_modes.append("multi-config (auto-detected)")

    if sum(sweep_modes) > 1:
        print("Error: Cannot use multiple sweep modes at the same time")
        print(f"Detected modes: {', '.join(active_modes)}")
        print(
            "Choose one of: --auto-sweep, --sweep-config, or define single configurations (not lists) in your config file"
        )
        sys.exit(1)

    # Handle allocation requests for different modes
    job_id = None
    if args.request_allocation:
        if args.auto_sweep:
            # For auto-sweep, use separate allocations for each configuration
            # This provides better resource utilization and fault tolerance
            print(
                "Auto-sweep mode: will request separate allocations for each configuration"
            )
            num_nodes = None  # No upfront allocation
        elif args.sweep_config and args.use_single_allocation:
            # For manual sweep with single allocation, calculate based on base config
            # This is a conservative estimate - actual sweep configs might need more
            base_nodes = calculate_nodes_needed(base_config, args.num_gpus)
            print(
                f"Manual sweep mode: using {base_nodes} nodes based on base configuration"
            )
            num_nodes = base_nodes
        elif not args.sweep_config and not is_multi_config:
            # For single configuration mode
            num_nodes = calculate_nodes_needed(base_config, args.num_gpus)
            print(
                f"Single config mode: calculated {num_nodes} nodes needed based on configuration"
            )
        else:
            # For manual sweep without single allocation or multi-config, no pre-allocation needed
            print(
                "Multi-config or manual sweep mode without single allocation - allocations will be requested per configuration"
            )
            num_nodes = None

        if num_nodes is not None:
            # Request the allocation
            job_id = get_slurm_allocation(args.account, args.partition,
                                          args.time, args.job_name, num_nodes)

            # Exit if allocation failed
            if not job_id:
                print("Exiting due to allocation failure.")
                sys.exit(1)

            # Set the SLURM_JOB_ID environment variable to ensure the jobs use this allocation
            os.environ["SLURM_JOB_ID"] = job_id
            print(f"Using SLURM allocation with job ID: {job_id}")

    try:
        # Auto-sweep mode: automatically determine optimal configurations
        if args.auto_sweep:
            print("=== Auto-Sweep Mode ===")
            auto_sweeper = AutoSweeper(args, max_gpus=args.max_gpus)
            auto_sweeper.run_auto_sweep()
        # Multi-config mode: test all combinations of predefined configurations (auto-detected)
        elif is_multi_config:
            print("=== Multi-Configuration Mode (Auto-Detected) ===")
            print(
                "Detected lists in configuration file - will test all combinations"
            )
            multi_sweeper.run_multi_config_sweep()
        # Manual sweep mode: use provided sweep config
        elif args.sweep_config:
            print("=== Manual Sweep Mode ===")
            sweeper = ParameterSweeper(args.sweep_config, args)
            sweeper.run_sweeps()
        else:
            # Standard single-run mode
            print("=== Single Configuration Mode ===")
            with JobManager(args) as job_manager:
                job_manager.launch_jobs()
    finally:
        # Release the allocation if we requested one and it's not a manual sweep (which handles its own allocations)
        if job_id and (not args.sweep_config
                       or args.use_single_allocation) and not is_multi_config:
            print(f"Releasing SLURM allocation (job ID: {job_id})...")
            subprocess.run(["scancel", job_id])
            print("SLURM allocation released.")


if __name__ == "__main__":
    main()
