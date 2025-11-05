"""Disaggregated Benchmark Executor.

Simplified version.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import GPU_RESOURCE_CONFIG, SESSION_COLLECT_CMD_TYPE, EnvManager, extract_config_fields
from report import LogParser, LogWritter, ResultSaver
from trt_test_alternative import call, check_output

# ============================================================================
# SLURM Run Command Builder
# ============================================================================


class SlurmRunCommandBuilder:
    """SLURM Run Command Builder.

    Build srun commands for different GPU types and command types.
    Reuses GPU_RESOURCE_CONFIG for consistency with SlurmJobBuilder.
    """

    def build_srun_prefix(self, job_name: str) -> List[str]:
        """Build srun command prefix based on GPU type."""
        gpu_type = EnvManager.get_gpu_type()

        # Reuse the same GPU_RESOURCE_CONFIG as SlurmJobBuilder
        gpu_config = GPU_RESOURCE_CONFIG.get(gpu_type)
        if not gpu_config:
            raise ValueError(
                f"GPU resource configuration not found for {gpu_type}. "
                f"Please add configuration in GPU_RESOURCE_CONFIG."
            )

        # Common srun arguments
        srun_args = [
            "srun",
            "-l",
            "--container-name=sysinfo-get",
            f"--container-image={EnvManager.get_container_image()}",
            f"--container-mounts={EnvManager.get_container_mount()}",
        ]

        # Add GPU-specific gres parameter (reuse gres_gpu field)
        # If gres_gpu is not None, add --gres parameter
        if gpu_config["gres_gpu"] is not None:
            srun_args.append(f"--gres=gpu:{gpu_config['gres_gpu']}")

        # Add common parameters
        srun_args.extend(
            [
                f"--partition={EnvManager.get_slurm_partition()}",
                f"--account={EnvManager.get_slurm_account()}",
                f"--job-name={job_name}",
                "--time=02:00:00",
                "--mpi=pmix",
                "--overlap",
                "-N",
                "1",
                "-n",
                "1",
            ]
        )

        return srun_args

    def build_script_command(self, cmd_type: str) -> List[str]:
        """Build script command based on command type."""
        work_dir = EnvManager.get_work_dir()
        output_path = EnvManager.get_output_path()
        install_mode = EnvManager.get_install_mode()
        repo_dir = EnvManager.get_repo_dir()

        if cmd_type == SESSION_COLLECT_CMD_TYPE:
            if install_mode == "none":
                return [
                    "bash",
                    "-c",
                    f"cd {work_dir} && python3 {work_dir}/simple_collect.py {output_path}",
                ]
            elif install_mode == "source":
                install_cmd = f"""
                cd {repo_dir}
                pip3 install -e . || echo '⚠️  Source install failed, continuing...'

                echo '✅ Source installation completed'

                echo '🚀 Step 3: Running simple_collect.py...'
                cd {work_dir}
                python3 {work_dir}/simple_collect.py {output_path}
                """
                return ["bash", "-c", install_cmd]
            else:
                raise ValueError(f"Invalid install mode: {install_mode}")
        else:
            # Future command types can be added here
            # elif cmd_type == "benchmark_collect":
            #     model_dir = EnvManager.get_model_dir()
            #     return [
            #         "bash", "-c",
            #         f"cd {work_dir} && python3 {work_dir}/benchmark_collect.py "
            #         f"--model-dir {model_dir} --output {output_path}"
            #     ]
            # elif cmd_type == "metrics_collect":
            #     return [
            #         "bash", "-c",
            #         f"cd {work_dir} && python3 {work_dir}/metrics_collect.py --config {work_dir}/config.yaml"
            #     ]
            raise ValueError(
                f"Unsupported command type: {cmd_type}. "
                f"Currently supported: {SESSION_COLLECT_CMD_TYPE}"
            )

    def run_job(self, cmd_type: str, job_name: str) -> Dict[str, Any]:
        """Execute srun job."""
        try:
            # Build complete command
            srun_prefix = self.build_srun_prefix(job_name)
            script_command = self.build_script_command(cmd_type)
            full_command = srun_prefix + script_command

            # Execute
            check_output(full_command, timeout=7200)  # 2 hours = 7200 seconds
            return {"status": True, "msg": "Job executed successfully"}
        except Exception as e:
            print(f"Job execution failed: {e}")
            return {"status": False, "msg": str(e)}


def make_slurm_run_command():
    """Create run command function (maintain interface compatibility)."""
    builder = SlurmRunCommandBuilder()
    return builder.run_job


class JobManager:
    """Job manager class."""

    @staticmethod
    def submit_job(test_config) -> tuple:
        """Submit job using submit.py with YAML config.

        Args:
            test_config: TestConfig object containing configuration

        Returns:
            tuple: (success: bool, job_id: str)
        """
        print("🚀 Submitting job using submit.py...")

        try:
            import re

            # Call submit.py with the config file
            submit_script = os.path.join(EnvManager.get_script_dir(), "submit.py")

            cmd = ["python3", submit_script, "-c", test_config.config_path]

            print(f"   Command: {' '.join(cmd)}")

            # Execute submission using check_output
            output = check_output(cmd, timeout=60)
            print(f"   Output: {output}")

            # Parse job ID from output
            if "Submitted batch job" in output:
                match = re.search(r"Submitted batch job (\d+)", output)
                if match:
                    job_id = match.group(1)
                    print(f"   ✅ Job submitted successfully: {job_id}")
                    return True, job_id

            print("   ❌ Unable to extract job ID from output")
            return False, ""

        except Exception as e:
            error_msg = str(e)
            # Extract stderr from CalledProcessError if available
            if hasattr(e, "stderr") and e.stderr:
                error_msg = e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr
            print(f"   ❌ Job submission exception: {error_msg}")
            return False, error_msg

    @staticmethod
    def check_result(
        job_id: str,
        test_config,
        timestamps: Optional[Dict[str, str]] = None,
        test_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check job execution result and generate report.

        High-level method that automatically extracts parameters from TestConfig,
        parses logs, generates performance reports, and saves results to CSV.

        Args:
            job_id: SLURM job ID
            test_config: TestConfig object containing configuration
            timestamps: Optional timestamps dict for the test case
            test_name: Optional test name for reporting

        Returns:
            Dict with 'success' status and other result information
        """
        # Extract parameters from YAML config using common utility
        config_data = test_config.config_data
        fields = extract_config_fields(config_data)

        # Extract fields for logging and result directory
        log_base = fields["log_base"]
        context_dir = fields["context_dir"]
        log_dir_name = log_base

        print(f"   📁 Log directory: {log_dir_name}")
        print(f"   📁 Context directory: {context_dir}")

        result_dir = os.path.join(EnvManager.get_script_dir(), log_dir_name, context_dir)
        # Call the internal implementation method
        return JobManager._check_job_result(
            job_id=job_id,
            benchmark_type=test_config.benchmark_type,
            config=config_data,
            metrics_config=test_config.metrics_config,
            model_name=test_config.model_name,
            result_dir=result_dir,
            timestamps=timestamps,
            test_name=test_name,
        )

    @staticmethod
    def check_job_status(job_id: str) -> str:
        """Check job status using sacct (works for all job states)."""
        try:
            # Use sacct to get job status - works for both running and completed jobs
            sacct_output = check_output(
                ["sacct", "-j", job_id, "--noheader", "--format=State", "-X"], timeout=30
            )
            if sacct_output.strip():
                return sacct_output.strip()
            else:
                # If sacct returns empty, job might be very new, wait a bit and try once more
                time.sleep(3)
                sacct_output = check_output(
                    ["sacct", "-j", job_id, "--noheader", "--format=State", "-X"], timeout=30
                )
                return sacct_output.strip() if sacct_output.strip() else "UNKNOWN"
        except Exception as e:
            print(f"Error checking job status with sacct: {e}")
            return "ERROR"

    @staticmethod
    def wait_for_completion(job_id: str, timeout: int = 3600) -> bool:
        """Wait for job completion."""
        start_time = time.time()

        # Wait for job to appear in system (initial delay)
        print(f"   ⏳ Waiting for job {job_id} to appear in system...")
        time.sleep(10)  # Initial wait for job to be scheduled

        last_status = None  # Track status changes
        while time.time() - start_time < timeout:
            status = JobManager.check_job_status(job_id)

            # Only print when status changes
            if status != last_status:
                print(f"   📊 Job {job_id} status changed: {status}")
                last_status = status

            # Check for terminal states - all mean the job is done
            if status in [
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "ERROR",
                "CANCELLED+",
            ] or ("error" in status.lower() and status != "ERROR"):
                if status == "COMPLETED":
                    print(f"   ✅ Job {job_id} completed successfully")
                else:
                    print(f"   ❌ Job {job_id} finished with status: {status}")
                return True  # Job finished (successfully or with failure)

            # For running states, don't print repeatedly - status change already printed above
            # Only log unexpected/unknown statuses
            if status not in ["RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "UNKNOWN"]:
                print(f"   🔍 Job {job_id} has unexpected status: {status}")

            time.sleep(30)  # check every 30 seconds

        print(f"   ⏰ Job {job_id} timeout after {timeout} seconds")
        return False  # timeout

    @staticmethod
    def cancel_job(job_id: str) -> bool:
        """Cancel job."""
        try:
            call(["scancel", job_id], timeout=30)
            print(f"   🛑 Job cancelled: {job_id}")
            return True
        except Exception as e:
            print(f"   ❌ Job cancellation failed: {e}")
            return False

    def get_log_and_yaml_files(directory: str) -> List[str]:
        """Retrieve the list of filenames (without path) in the specified directory.

        That end with '.log', '.yaml', or '.yml'.

        Args:
            directory (str): Path to the target directory.

        Returns:
            List[str]: Sorted list of matching filenames.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path exists but is not a directory.
        """
        path = Path(directory)

        # Check if the path exists
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Check if the path is a directory
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        # Collect filenames with .log, .yaml, or .yml extensions
        files = [
            f.name for f in path.iterdir() if f.is_file() and f.suffix in [".log", ".yaml", ".yml"]
        ]

        # Return sorted list for consistent output
        return sorted(files)

    @staticmethod
    def _print_logs_to_console(job_id: str, result_dir: str) -> None:
        """Print SLURM log and all .log/.yaml files in result_dir to console.

        Args:
            job_id: SLURM job ID for finding the slurm log file
            result_dir: Result directory containing log and config files
        """
        # Print the slurm log to console
        slurm_log_writer = LogWritter(EnvManager.get_work_dir())
        slurm_log_writer.print_to_console(f"slurm-{job_id}.out")

        # Print all .log and .yaml files in result_dir (except output_server.log)
        log_writer = LogWritter(result_dir)
        files_to_print = []
        if os.path.exists(result_dir):
            for file in os.listdir(result_dir):
                if (
                    file.endswith(".log") or file.endswith(".yaml")
                ) and file != "output_server.log":
                    files_to_print.append(file)

        # Sort files for consistent output order
        files_to_print.sort()

        for file in files_to_print:
            if os.path.exists(os.path.join(result_dir, file)):
                log_writer.print_to_console(file)
            else:
                print(f"   ⚠️  {file} not found: {file}")

    @staticmethod
    def _check_job_result(
        job_id: str,
        benchmark_type: str,
        config: dict,
        metrics_config,
        model_name: str,
        result_dir: str,
        timestamps: Optional[Dict[str, str]] = None,
        test_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Internal method: Check job result with metrics config.

        This is a low-level method that requires manual parameter extraction.
        Use check_result() for a high-level interface with TestConfig.

        Args:
            job_id: SLURM job ID
            benchmark_type: Benchmark type (1k1k, 8k1k, etc.)
            config: Configuration dict (YAML data)
            metrics_config: MetricsConfig object (default or custom)
            model_name: Model name
            result_dir: Result directory
            timestamps: Optional timestamps dict
            test_name: Optional test name
        """
        result = {"job_id": job_id, "status": "UNKNOWN", "success": False}
        print(f"   📁 Checking result directory: {result_dir}")

        # Print logs and config files to console
        JobManager._print_logs_to_console(job_id, result_dir)

        # Parse using metrics config
        log_parser = LogParser(benchmark_type, config, metrics_config, result_dir)
        parse_result = log_parser.parse(model_name, timestamps=timestamps, test_name=test_name)

        if not parse_result["status"]:
            return result

        output_path = EnvManager.get_output_path()
        os.makedirs(output_path, exist_ok=True)

        output_csv = os.path.join(output_path, "perf_script_test_results.csv")
        result_saver = ResultSaver(output_csv)
        result_df = parse_result["df"]
        result_saver.append_a_df(result_df)
        result["success"] = True
        result["status"] = "SUCCESS"
        return result


# create executor function
run_job = make_slurm_run_command()
