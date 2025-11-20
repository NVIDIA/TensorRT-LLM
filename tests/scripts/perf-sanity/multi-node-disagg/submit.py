import argparse
import os
import subprocess
import sys
import time

import yaml


def read_yaml_config(config_path):
    """Read and parse the yaml configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def calculate_resources(config):
    """Calculate resources from the hardware config in disagg_configs."""
    hardware = config.get("hardware", {})

    num_ctx_servers = hardware.get("num_ctx_servers")
    num_gen_servers = hardware.get("num_gen_servers")
    gpus_per_node = hardware.get("gpus_per_node")
    gpus_per_ctx_server = hardware.get("gpus_per_ctx_server")
    gpus_per_gen_server = hardware.get("gpus_per_gen_server")

    if None in [
        num_ctx_servers,
        num_gen_servers,
        gpus_per_node,
        gpus_per_ctx_server,
        gpus_per_gen_server,
    ]:
        raise ValueError("Missing required hardware configuration in disagg config")

    # Calculate nodes per server
    nodes_per_ctx_server = (gpus_per_ctx_server + gpus_per_node - 1) // gpus_per_node
    nodes_per_gen_server = (gpus_per_gen_server + gpus_per_node - 1) // gpus_per_node

    total_nodes = num_ctx_servers * nodes_per_ctx_server + num_gen_servers * nodes_per_gen_server
    total_gpus = total_nodes * gpus_per_node

    return {
        "num_ctx_servers": num_ctx_servers,
        "num_gen_servers": num_gen_servers,
        "gpus_per_node": gpus_per_node,
        "gpus_per_ctx_server": gpus_per_ctx_server,
        "gpus_per_gen_server": gpus_per_gen_server,
        "nodes_per_ctx_server": nodes_per_ctx_server,
        "nodes_per_gen_server": nodes_per_gen_server,
        "total_nodes": total_nodes,
        "total_gpus": total_gpus,
    }


def main():
    parser = argparse.ArgumentParser(description="Submit multinode disagg perf tests to SLURM")
    parser.add_argument("--partition", required=True, help="SLURM Partition")
    parser.add_argument("--jobname", required=True, help="SLURM Job name")
    parser.add_argument("--account", required=True, help="SLURM Account")
    parser.add_argument("--trtllmsrc", required=True, help="TRT-LLM repo source path")
    parser.add_argument("--jobworkspace", required=True, help="Job workspace directory")
    parser.add_argument(
        "--stagename",
        default="GB200-8_GPUs-2_Nodes-PyTorch-Perf-Sanity-Disagg-Post-Merge-1",
        help="Stage name (optional)",
    )
    parser.add_argument(
        "--test-name",
        required=True,
        dest="test_name",
        help="Test name (e.g., l0_gb200_multi_nodes_disagg-r1_fp4_v2_dep8_mtp1)",
    )
    parser.add_argument("--mounts", required=True, help="Container mounts")
    parser.add_argument(
        "--llm-models-root",
        default="/home/scratch.trt_llm_data/llm-models",
        dest="llm_models_root",
        help="LLM models root directory",
    )
    parser.add_argument(
        "--build-wheel",
        action="store_true",
        dest="build_wheel",
        help="Build TensorRT-LLM wheel before running tests",
    )

    args = parser.parse_args()

    # Extract config file name from test_name
    config_name = args.test_name.split("-")[0]
    config_path = os.path.join(args.trtllmsrc, "tests/scripts/perf-sanity", f"{config_name}.yaml")
    config = read_yaml_config(config_path)
    resources = calculate_resources(config)
    print(f"Resources: {resources}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    launch_sh = os.path.join(script_dir, "launch.sh")
    os.makedirs(args.jobworkspace, exist_ok=True)

    sbatch_cmd = [
        "sbatch",
        f"--nodes={resources['total_nodes']}",
        f"--ntasks={resources['total_gpus']}",
        f"--ntasks-per-node={resources['gpus_per_node']}",
        f"--gpus-per-node={resources['gpus_per_node']}",
        f"--partition={args.partition}",
        "--time=04:00:00",
        f"--account={args.account}",
        f"-J={args.jobname}",
        f"-o {args.jobworkspace}/slurm.out.log",
        f"-e {args.jobworkspace}/slurm.error.log",
        launch_sh,
        str(resources["num_ctx_servers"]),
        str(resources["num_gen_servers"]),
        str(resources["gpus_per_node"]),
        str(resources["gpus_per_ctx_server"]),
        str(resources["gpus_per_gen_server"]),
        str(resources["nodes_per_ctx_server"]),
        str(resources["nodes_per_gen_server"]),
        str(resources["total_nodes"]),
        str(resources["total_gpus"]),
        args.trtllmsrc,
        args.jobworkspace,
        script_dir,
        args.stagename,
        args.test_name,
        args.mounts,
        args.llm_models_root,
        "true" if args.build_wheel else "false",
    ]

    print("\nSubmitting job with command:")
    print(" ".join(sbatch_cmd))

    # Submit the job
    try:
        result = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)
        print(f"\n{result.stdout}")
        if result.stderr:
            print(f"stderr: {result.stderr}", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}", file=sys.stderr)
        print(f"stdout: {e.stdout}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        sys.exit(1)

    # Extract job ID
    job_id = None
    for line in result.stdout.split("\n"):
        if "Submitted batch job" in line:
            job_id = line.split()[-1]
            break
    if not job_id:
        print("Error: Could not extract job ID from sbatch output")
        sys.exit(1)
    print(f"Submitted job {job_id}")

    # Create and tail the output file
    output_file = os.path.join(args.jobworkspace, "slurm_output.log")
    subprocess.run(["touch", output_file], check=False)
    tail_proc = subprocess.Popen(
        ["tail", "-f", output_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    tail_pid = tail_proc.pid
    print(f"Started tail process with PID {tail_pid}")

    # Wait until sbatch job is done
    print("Monitoring job status...")
    while True:
        check_cmd = f"squeue -j {job_id} -o %T"
        check_result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
        if check_result.returncode != 0:
            print("Job is no longer in queue")
            break
        time.sleep(60)

    print("Terminating tail process...")
    tail_proc.terminate()
    tail_proc.wait()

    # Check if the job failed or not
    print("Checking job exit code...")
    sacct_cmd = f"sacct -j {job_id} --format=ExitCode -Pn --allocations"
    sacct_result = subprocess.run(sacct_cmd, shell=True, capture_output=True, text=True)
    exit_code_str = sacct_result.stdout.strip()
    exit_code = exit_code_str.split(":")[0] if ":" in exit_code_str else "1"
    exit_code = int(exit_code) if exit_code.isdigit() else 1
    if exit_code != 0:
        print(f"Pytest failed in Slurm job {job_id} with exit code {exit_code}")
        sys.exit(exit_code)
    print("Job completed successfully")


if __name__ == "__main__":
    main()
