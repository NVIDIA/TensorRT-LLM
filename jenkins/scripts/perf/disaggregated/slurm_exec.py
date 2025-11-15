#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time

import yaml


def main():
    parser = argparse.ArgumentParser(description="Execute SLURM disaggregated server tests")
    parser.add_argument("--sbatch-params", required=True, help="Additional sbatch parameters")
    parser.add_argument("--config-yaml", required=True, help="Path to config YAML file")
    parser.add_argument("--slurm-launch-sh", required=True, help="Path to slurm_launch.sh script")
    parser.add_argument("--job-workspace", required=True, help="Job workspace directory")
    parser.add_argument("--llm-tarfile", required=True, help="LLM tarfile URL")
    parser.add_argument("--tar-name", required=True, help="Tar filename")
    parser.add_argument("--llm-src-node", required=True, help="LLM source path on node")
    parser.add_argument("--stage-name", required=True, help="Stage name")
    parser.add_argument("--perf-mode", required=True, help="Performance mode flag")
    parser.add_argument("--resource-path-node", required=True, help="Resource path on node")
    parser.add_argument("--pytest-command", required=True, help="Pytest command")
    parser.add_argument("--coverage-config-file", required=True, help="Coverage config file path")
    parser.add_argument("--container", required=True, help="Container image")
    parser.add_argument("--mounts", required=True, help="Container mounts")
    parser.add_argument("--script-run-node", required=True, help="Path to slurm_run.sh on node")
    parser.add_argument("--script-install-node", required=True, help="Path to install.sh on node")
    parser.add_argument("--test-list-path-node", required=True, help="Path to test list on node")
    parser.add_argument("--output-path", required=True, help="Output log path")

    args = parser.parse_args()

    # Parse config yaml to get hardware configs
    with open(args.config_yaml, "r") as f:
        config = yaml.safe_load(f)

    # Get hardware configuration from disagg_configs
    disagg_configs = config.get("disagg_configs", [])
    if not disagg_configs:
        print("Error: No disagg_configs found in YAML file")
        sys.exit(1)

    # Get hardware from first config entry
    hardware = None
    for item in disagg_configs:
        if "hardware" in item:
            hardware = item["hardware"]
            break

    if not hardware:
        print("Error: No hardware configuration found in disagg_configs")
        sys.exit(1)

    num_ctx_servers = hardware.get("num_ctx_servers", 1)
    num_gen_servers = hardware.get("num_gen_servers", 1)
    gpus_per_node = hardware.get("gpus_per_node", 1)
    gpus_per_ctx_server = hardware.get("gpus_per_ctx_server", 1)
    gpus_per_gen_server = hardware.get("gpus_per_gen_server", 1)

    # Calculate nodes per server
    nodes_per_ctx_server = (gpus_per_ctx_server + gpus_per_node - 1) // gpus_per_node
    nodes_per_gen_server = (gpus_per_gen_server + gpus_per_node - 1) // gpus_per_node

    total_nodes = num_ctx_servers * nodes_per_ctx_server + num_gen_servers * nodes_per_gen_server
    total_gpus = total_nodes * gpus_per_node

    # Build sbatch command
    sbatch_cmd = f"""sbatch \\
--nodes={total_nodes} \\
--ntasks={total_gpus} \\
--ntasks-per-node={gpus_per_node} \\
--gpus-per-node={gpus_per_node} \\
{args.sbatch_params.strip()} \\
--job-name=disagg-test-{args.job_workspace.split("/")[-1]} \\
{args.slurm_launch_sh} \\
"{args.job_workspace}" \\
"{args.llm_tarfile}" \\
"{args.tar_name}" \\
"{args.llm_src_node}" \\
"{args.stage_name}" \\
"{args.perf_mode}" \\
"{args.resource_path_node}" \\
"{args.pytest_command}" \\
"{args.coverage_config_file}" \\
"{args.container}" \\
"{args.mounts}" \\
"{args.script_run_node}" \\
"{args.script_install_node}" \\
"{args.config_yaml}" \\
"{args.test_list_path_node}" \\
"{num_ctx_servers}" \\
"{num_gen_servers}" \\
"{gpus_per_node}" \\
"{gpus_per_ctx_server}" \\
"{gpus_per_gen_server}" \\
"{nodes_per_ctx_server}" \\
"{nodes_per_gen_server}" \\
"{total_nodes}" \\
"{total_gpus}"
"""

    print(f"Running sbatch command:\n{sbatch_cmd}")
    result = subprocess.run(sbatch_cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"Error: sbatch failed with return code {result.returncode}")
        sys.exit(result.returncode)

    # Extract job ID from output
    job_id = None
    for line in result.stdout.split("\n"):
        if "Submitted batch job" in line:
            job_id = line.split()[-1]
            break

    if not job_id:
        print("Error: Could not extract job ID from sbatch output")
        sys.exit(1)

    print(f"Submitted job {job_id}")

    # Tail the output file
    output_file = args.output_path
    subprocess.run(["touch", output_file], check=False)
    tail_proc = subprocess.Popen(
        ["tail", "-f", output_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    tail_pid = tail_proc.pid
    print(f"Started tail process with PID {tail_pid}")

    try:
        # Wait until sbatch job is done
        print("Monitoring job status...")
        while True:
            check_cmd = f"squeue -j {job_id} -o %T"
            check_result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
            if check_result.returncode != 0:
                print("Job is no longer in queue")
                break
            time.sleep(300)

        # Kill tail -f process
        print("Terminating tail process...")
        tail_proc.terminate()
        try:
            tail_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tail_proc.kill()
            tail_proc.wait()

        # Check if the job failed or not
        print("Checking job exit code...")
        sacct_cmd = f"sacct -j {job_id} --format=ExitCode -Pn --allocations"
        sacct_result = subprocess.run(sacct_cmd, shell=True, capture_output=True, text=True)
        exit_code_str = sacct_result.stdout.strip()
        exit_code = exit_code_str.split(":")[0] if ":" in exit_code_str else "1"
        exit_code = int(exit_code) if exit_code.isdigit() else 1

        print(f"Job exit code: {exit_code}")

        if exit_code != 0:
            print(f"Pytest failed in Slurm job {job_id} with exit code {exit_code}")
            sys.exit(exit_code)

        print("Job completed successfully")
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, cleaning up...")
        tail_proc.terminate()
        try:
            tail_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tail_proc.kill()
            tail_proc.wait()
        sys.exit(1)
    except Exception as e:
        print(f"Error during job monitoring: {e}")
        tail_proc.terminate()
        try:
            tail_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tail_proc.kill()
            tail_proc.wait()
        sys.exit(1)


if __name__ == "__main__":
    main()
