import argparse
import os

from disagg_profiler.job_manager import JobManager, get_slurm_allocation


def setup_argparse():
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="TensorRT-LLM Disaggregated Serving Launcher")

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
    parser.add_argument("--num-nodes",
                        type=int,
                        required=True,
                        help="Number of nodes")
    parser.add_argument("--ntasks",
                        type=int,
                        required=True,
                        help="Number of tasks")
    parser.add_argument("--ntasks-per-node",
                        type=int,
                        required=True,
                        help="Number of tasks per node")

    parser.add_argument("--trtllm-repo",
                        type=str,
                        default=None,
                        help="TensorRT-LLM repository")
    parser.add_argument("--config-file",
                        type=str,
                        default="./config.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--experiment-path",
                        type=str,
                        default=".",
                        help="Path to the configuration file")
    parser.add_argument("--num-disagg-servers",
                        type=int,
                        default=1,
                        help="Number of disaggregated servers to use.")

    return parser


def main():
    parser = setup_argparse()
    args = parser.parse_args()

    slurm_job_id = get_slurm_allocation(args.account,
                                        args.partition,
                                        args.time,
                                        f"{args.job_name}",
                                        num_nodes=args.num_nodes)
    os.environ["SLURM_JOB_ID"] = slurm_job_id
    with JobManager(args) as job_manager:
        job_manager.launch_jobs()


if __name__ == "__main__":
    main()
