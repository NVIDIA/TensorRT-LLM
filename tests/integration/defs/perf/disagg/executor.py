"""Disaggregated Benchmark Executor.

Simplified version.
"""

import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional

from common import (DEBUG_MODE, GPU_RESOURCE_CONFIG, SESSION_COLLECT_CMD_TYPE,
                    EnvManager, extract_config_fields)
from report import LogParser, LogWriter, ResultSaver
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
                f"Please add configuration in GPU_RESOURCE_CONFIG.")

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
        srun_args.extend([
            f"--partition={EnvManager.get_slurm_partition()}",
            f"--account={EnvManager.get_slurm_account()}",
            f"--job-name={job_name}",
            "--time=02:00:00",
            "--mpi=pmix",
            # Note: Removed --overlap to ensure GPU allocation for session_collect
            # which runs after all test jobs have completed
            "-N",
            "1",
            "-n",
            "1",
        ])

        return srun_args

    def build_script_command(self, cmd_type: str) -> List[str]:
        """Build script command based on command type."""
        work_dir = EnvManager.get_work_dir()
        output_path = EnvManager.get_output_path()
        install_mode = EnvManager.get_install_mode()
        repo_dir = EnvManager.get_repo_dir()
        trtllm_wheel_path = EnvManager.get_trtllm_wheel_path()

        if cmd_type == SESSION_COLLECT_CMD_TYPE:
            if install_mode == "none":
                return [
                    "bash",
                    "-c",
                    f"cd {work_dir} && python3 {work_dir}/simple_collect.py {output_path}",
                ]
            elif install_mode == "wheel":
                # Install TensorRT-LLM wheel first, then run simple_collect.py
                # Note: Use --no-deps to avoid overwriting container's pre-installed packages (like torch)
                install_cmd = f"""
                    cd {repo_dir}
                    echo '📦 Step 1: Installing TensorRT-LLM wheel...'
                    pip3 install {trtllm_wheel_path} || echo '⚠️  Wheel install failed, continuing...'
                    echo '✅ Wheel installation completed'

                    echo '🚀 Step 2: Running simple_collect.py...'
                    cd {work_dir}
                    python3 {work_dir}/simple_collect.py {output_path}
                """
                return ["bash", "-c", install_cmd]
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
            raise ValueError(f"Unsupported command type: {cmd_type}. "
                             f"Currently supported: {SESSION_COLLECT_CMD_TYPE}")

    def run_job(self,
                cmd_type: str,
                job_name: str,
                log_file: str = None) -> Dict[str, Any]:
        """Execute srun job.

        Args:
            cmd_type: Type of command to execute
            job_name: Name for the SLURM job
            log_file: Optional path to save command output

        Returns:
            Dict with status and message
        """
        try:
            # Build complete command
            srun_prefix = self.build_srun_prefix(job_name)
            script_command = self.build_script_command(cmd_type)
            full_command = srun_prefix + script_command

            # Execute with optional log file
            if log_file:
                print(f"   📝 Saving output to: {log_file}")
                # Use Python file redirection to avoid shell quoting issues
                import subprocess

                with open(log_file, "w") as f:
                    result = subprocess.run(full_command,
                                            stdout=f,
                                            stderr=subprocess.STDOUT,
                                            timeout=7200,
                                            text=True)
                    if result.returncode != 0:
                        raise subprocess.CalledProcessError(
                            result.returncode, full_command)
                print(f"   ✅ Output saved to {log_file}")
                output = ""  # Output is in file
            else:
                output = check_output(full_command, timeout=7200)

            return {
                "status": True,
                "msg": "Job executed successfully",
                "output": output
            }
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
            submit_script = os.path.join(EnvManager.get_script_dir(),
                                         "submit.py")

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
                error_msg = e.stderr.decode() if isinstance(e.stderr,
                                                            bytes) else e.stderr
            print(f"   ❌ Job submission exception: {error_msg}")
            return False, error_msg

    @staticmethod
    def backup_logs(job_id: str, test_config, result_dir: str,
                    is_passed: bool) -> Optional[str]:
        """Backup logs and config files to test_id directory.

        Args:
            job_id: SLURM job ID
            test_config: TestConfig object
            result_dir: Result directory path
            is_passed: Whether the job passed
        Returns:
            backup_dir path if successful, None otherwise
        """
        # Copy result_dir to a timestamped backup directory
        if os.path.exists(result_dir):
            # Replace colons with underscores for safe directory naming
            dst_dir_name = test_config.test_id.replace(":", "-")
            # Add ERROR suffix if the job failed
            if not is_passed:
                dst_dir_name = f"{dst_dir_name}_ERROR"
            backup_dir = os.path.join(os.path.dirname(result_dir), dst_dir_name)

            try:
                print("   📦 Copying result directory to backup...")
                print(f"   📁 Source: {result_dir}")
                print(f"   📁 Destination: {backup_dir}")

                # Remove old backup if it exists
                if os.path.exists(backup_dir):
                    print(
                        "   ⚠️  Warning: Backup directory already exists, removing old backup"
                    )
                    shutil.rmtree(backup_dir)

                shutil.copytree(result_dir, backup_dir)
                print(f"   ✅ Backup created successfully: {backup_dir}")

                work_dir = EnvManager.get_work_dir()
                slurm_out_file = os.path.join(work_dir, f"slurm-{job_id}.out")
                if os.path.exists(slurm_out_file):
                    shutil.copy(slurm_out_file, backup_dir)
                    print(
                        f"   ✅ SLURM log copied successfully: {slurm_out_file}")
                else:
                    print(
                        f"   ⚠️  Warning: SLURM log not found: {slurm_out_file}"
                    )

                case_config_path = test_config.config_path
                if os.path.exists(case_config_path):
                    shutil.copy(case_config_path, backup_dir)
                    print(
                        f"   ✅ Case config copied successfully: {case_config_path}"
                    )
                else:
                    print(
                        f"   ⚠️  Warning: Case config not found: {case_config_path}"
                    )

                return backup_dir
            except Exception as e:
                print(f"   ⚠️  Warning: Failed to create backup copy: {e}")
                return None
        else:
            print(
                f"   ⚠️  Warning: Result directory does not exist yet: {result_dir}"
            )
            return None

    @staticmethod
    def cleanup_result_dir(result_dir: str) -> bool:
        """Clean up result directory.

        Args:
            result_dir: Result directory path

        Returns:
            True if successful, False otherwise
        """
        if os.path.exists(result_dir):
            try:
                shutil.rmtree(result_dir)
                print(f"   ✅ Result directory removed: {result_dir}")
                return True
            except Exception as e:
                print(f"   ⚠️  Warning: Failed to remove result directory: {e}")
                return False
        return True

    @staticmethod
    def get_result_dir(test_config) -> str:
        """Get result directory.

        Args:
            test_config: TestConfig object

        Returns:
            Result directory path
        """
        config_data = test_config.config_data
        fields = extract_config_fields(config_data)

        # Extract fields for logging and result directory
        log_base = fields["log_base"]
        context_dir = fields["context_dir"]
        log_dir_name = log_base

        print(f"   📁 Log directory: {log_dir_name}")
        print(f"   📁 Context directory: {context_dir}")

        result_dir = os.path.join(EnvManager.get_script_dir(), log_dir_name,
                                  context_dir)
        return result_dir

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
        config_data = test_config.config_data
        # Get result directory
        result_dir = JobManager.get_result_dir(test_config)

        # Call the internal implementation method
        check_result = JobManager._check_job_result(
            job_id=job_id,
            test_category=test_config.
            test_category,  # Pass test category for routing
            benchmark_type=test_config.benchmark_type,
            config=config_data,
            metrics_config=test_config.metrics_config,
            accuracy_config=test_config.accuracy_config,  # Pass accuracy config
            model_name=test_config.model_name,
            result_dir=result_dir,
            timestamps=timestamps,
            test_name=test_name,
        )

        is_passed = check_result["success"]
        # Backup logs and config files
        JobManager.backup_logs(job_id, test_config, result_dir, is_passed)

        # Clean up result directory
        if DEBUG_MODE:
            print(
                f"🐛 Debug mode: Skipping result directory cleanup: {result_dir}"
            )
        else:
            JobManager.cleanup_result_dir(result_dir)

        return check_result

    @staticmethod
    def check_for_early_failure(job_id: str,
                                test_config) -> tuple[bool, Optional[str]]:
        """Check logs for early failure indicators.

        Args:
            job_id: SLURM job ID
            test_config: TestConfig object

        Returns:
            tuple: (has_error, error_message)
        """
        # Key error patterns
        error_patterns = [
            (
                r"\[E\]\s+Traceback[\s\S]*?mpi4py\.MPI\.Comm\.allgather",
                "MPI communication error detected",
            ),
            (
                r"\[E\]\s+Traceback[\s\S]*?pickle data was truncated",
                "Pickle serialization error detected",
            ),
        ]

        try:
            # Check gen and ctx server logs if result_dir exists
            try:
                result_dir = JobManager.get_result_dir(test_config)
                if os.path.exists(result_dir):
                    # Find all output_gen_*.log and output_ctx_*.log files
                    for filename in os.listdir(result_dir):
                        is_gen_log = filename.startswith("output_gen_")
                        is_ctx_log = filename.startswith("output_ctx_")
                        if (is_gen_log
                                or is_ctx_log) and filename.endswith(".log"):
                            log_path = os.path.join(result_dir, filename)
                            try:
                                with open(log_path,
                                          "r",
                                          encoding="utf-8",
                                          errors="ignore") as f:
                                    # Read last 100KB
                                    f.seek(0, 2)
                                    file_size = f.tell()
                                    f.seek(max(0, file_size - 102400), 0)
                                    recent_content = f.read()

                                    for pattern, error_msg in error_patterns:
                                        if re.search(pattern, recent_content,
                                                     re.MULTILINE):
                                            return True, f"{error_msg} in {filename}"
                            except Exception as e:
                                print(
                                    f"   ⚠️  Warning: Failed to check {filename}: {e}"
                                )
            except Exception:
                # result_dir might not exist yet, that's OK
                pass

        except Exception as e:
            print(f"   ⚠️  Warning: Error during early failure check: {e}")

        return False, None

    @staticmethod
    def check_job_status(job_id: str) -> str:
        """Check job status using sacct (works for all job states)."""
        try:
            # Use sacct to get job status - works for both running and completed jobs
            sacct_output = check_output(
                ["sacct", "-j", job_id, "--noheader", "--format=State", "-X"],
                timeout=30)
            if sacct_output.strip():
                return sacct_output.strip()
            else:
                # If sacct returns empty, job might be very new, wait a bit and try once more
                time.sleep(3)
                sacct_output = check_output([
                    "sacct", "-j", job_id, "--noheader", "--format=State", "-X"
                ],
                                            timeout=30)
                return sacct_output.strip() if sacct_output.strip(
                ) else "UNKNOWN"
        except Exception as e:
            print(f"Error checking job status with sacct: {e}")
            return "ERROR"

    @staticmethod
    def wait_for_completion(
            job_id: str,
            timeout: int = 3600,
            test_config=None,
            check_early_failure: bool = True) -> tuple[bool, Optional[str]]:
        """Wait for job completion with optional early failure detection.

        Args:
            job_id: SLURM job ID
            timeout: Maximum wait time in seconds
            test_config: TestConfig object (required for early failure detection)
            check_early_failure: Whether to check logs for early failures

        Returns:
            tuple: (completed_successfully, error_message)
                - (True, None): Job completed normally
                - (False, "timeout"): Job timed out
                - (False, error_msg): Job failed early with specific error
        """
        start_time = time.time()
        check_interval = 180  # Check every 3 minutes
        failure_check_interval = 60  # Check for failures every 60 seconds
        last_failure_check = start_time

        # Wait for job to appear in system (initial delay)
        print(f"   ⏳ Waiting for job {job_id} to appear in system...")
        time.sleep(60)  # Initial wait for job to be scheduled

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
                    return True, None
                else:
                    print(f"   ❌ Job {job_id} finished with status: {status}")
                    return True, None  # Job finished (let check_result determine success)

            # For running states, don't print repeatedly - status change already printed above
            # Only log unexpected/unknown statuses
            if status not in [
                    "RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "UNKNOWN"
            ]:
                print(f"   🔍 Job {job_id} has unexpected status: {status}")

            # Check for early failures (only when job is running and test_config is provided)
            current_time = time.time()
            if (check_early_failure and test_config and status == "RUNNING"
                    and current_time - last_failure_check
                    >= failure_check_interval):
                has_error, error_msg = JobManager.check_for_early_failure(
                    job_id, test_config)
                if has_error:
                    print(f"   🚨 Early failure detected: {error_msg}")
                    print(f"   🛑 Stopping wait for job {job_id}")
                    return False, error_msg
                last_failure_check = current_time

            time.sleep(check_interval)

        print(f"   ⏰ Job {job_id} timeout after {timeout} seconds")
        return False, "timeout"

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

    @staticmethod
    def _print_logs_to_console(job_id: str, result_dir: str) -> None:
        """Print SLURM log and all .log/.yaml files in result_dir to console.

        Args:
            job_id: SLURM job ID for finding the slurm log file
            result_dir: Result directory containing log and config files
        """
        # Print the slurm log to console
        slurm_log_writer = LogWriter(EnvManager.get_work_dir())
        slurm_log_writer.print_to_console(f"slurm-{job_id}.out")

        # Print all .log and .yaml files in result_dir (except output_server.log)
        log_writer = LogWriter(result_dir)
        files_to_print = []
        if os.path.exists(result_dir):
            for file in os.listdir(result_dir):
                if (file.endswith(".log") or
                        file.endswith(".yaml")) and file != "output_server.log":
                    files_to_print.append(file)

        # Sort files for consistent output order
        files_to_print.sort()

        for file in files_to_print:
            if os.path.exists(os.path.join(result_dir, file)):
                log_writer.print_to_console(file)
            else:
                print(f"   ⚠️  {file} not found: {file}")

    @staticmethod
    def _check_accuracy_result(
        job_id: str,
        metrics_config,
        accuracy_config,
        result_dir: str,
    ) -> Dict[str, Any]:
        """Check accuracy test result.

        Args:
            job_id: SLURM job ID
            metrics_config: MetricsConfig object
            accuracy_config: AccuracyConfig object
            result_dir: Result directory

        Returns:
            Dict with success status and accuracy details
        """
        # Initialize base result
        result: Dict[str, Any] = {
            "job_id": job_id,
            "status": "UNKNOWN",
            "success": False
        }

        # Validate accuracy_config
        if not accuracy_config:
            result["error"] = "Accuracy config not found in test configuration"
            return result

        # Import and use AccuracyParser
        from accuracy_parser import AccuracyParser

        accuracy_parser = AccuracyParser(metrics_config, accuracy_config,
                                         result_dir)
        validation_result = accuracy_parser.parse_and_validate()

        # Check if parsing succeeded
        if not validation_result["success"]:
            result["error"] = validation_result.get(
                "error", "Accuracy validation failed")
            return result

        # Print validation results
        print(f"   📊 Accuracy Validation Results:")
        all_passed = validation_result["all_passed"]

        # Print results for each run (using dataclass attributes for type safety)
        for run_validation in validation_result.get("runs", []):
            run_name = run_validation.run_name
            run_passed = run_validation.all_passed
            run_icon = "✅" if run_passed else "❌"

            print(f"   {run_icon} {run_name}:")

            for ds_result in run_validation.results:
                status_icon = "✅" if ds_result.passed else "❌"
                dataset_name = ds_result.dataset
                filter_type = ds_result.filter
                threshold_type = ds_result.threshold_type

                print(
                    f"      {status_icon} {dataset_name} ({filter_type}) - {threshold_type}:"
                )
                if ds_result.error:
                    print(f"         ⚠️  Error: {ds_result.error}")
                else:
                    print(f"         Expected: {ds_result.expected:.4f}")
                    print(f"         Actual:   {ds_result.actual:.4f}")
                    print(
                        f"         Threshold: {ds_result.threshold} ({ds_result.threshold_type})"
                    )
                    print(f"         {ds_result.message}")

        # Set result status
        if all_passed:
            print(f"   ✅ All accuracy tests PASSED (all runs)")
            result["success"] = True
            result["status"] = "PASSED"
        else:
            print(f"   ❌ Some accuracy tests FAILED")
            result["success"] = False
            result["status"] = "FAILED"
            result["error"] = "Some accuracy tests FAILED"

        # Add detailed results
        result["all_passed"] = validation_result["all_passed"]
        result["accuracy_runs"] = validation_result["runs"]
        result["raw_accuracy"] = validation_result.get("raw_results", [])

        return result

    @staticmethod
    def _check_perf_result(
        job_id: str,
        benchmark_type: str,
        config: dict,
        metrics_config,
        model_name: str,
        result_dir: str,
        timestamps: Optional[Dict[str, str]] = None,
        test_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check performance test result.

        Args:
            job_id: SLURM job ID
            benchmark_type: Benchmark type (1k1k, 8k1k, etc.)
            config: Configuration dict (YAML data)
            metrics_config: MetricsConfig object
            model_name: Model name
            result_dir: Result directory
            timestamps: Optional timestamps dict
            test_name: Optional test name

        Returns:
            Dict with success status and performance details
        """
        result = {"job_id": job_id, "status": "UNKNOWN", "success": False}

        # Parse metrics and save to CSV
        log_parser = LogParser(benchmark_type, config, metrics_config,
                               result_dir)
        parse_result = log_parser.parse(model_name,
                                        timestamps=timestamps,
                                        test_name=test_name)

        if not parse_result["status"]:
            return result

        # Check if df is None
        result_df = parse_result.get("df")
        if result_df is None:
            print("   ❌ Parse result contains None DataFrame")
            return result

        # Save results to CSV
        output_path = EnvManager.get_output_path()
        os.makedirs(output_path, exist_ok=True)

        output_csv = os.path.join(output_path, "perf_script_test_results.csv")
        result_saver = ResultSaver(output_csv)
        result_saver.append_a_df(result_df)

        result["success"] = True
        result["status"] = "SUCCESS"
        return result

    @staticmethod
    def _check_job_result(
        job_id: str,
        test_category: str,
        benchmark_type: str,
        config: dict,
        metrics_config,
        accuracy_config,
        model_name: str,
        result_dir: str,
        timestamps: Optional[Dict[str, str]] = None,
        test_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Internal method: Check job result with category routing.

        This is a low-level method that requires manual parameter extraction.
        Use check_result() for a high-level interface with TestConfig.

        Args:
            job_id: SLURM job ID
            test_category: Test category ("perf" or "accuracy")
            benchmark_type: Benchmark type (1k1k, 8k1k, etc.)
            config: Configuration dict (YAML data)
            metrics_config: MetricsConfig object (default or custom)
            accuracy_config: AccuracyConfig object (required for accuracy tests)
            model_name: Model name
            result_dir: Result directory
            timestamps: Optional timestamps dict
            test_name: Optional test name

        Returns:
            Dict with success status and details
        """
        print(f"   📁 Checking result directory: {result_dir}")

        # Print logs and config files to console
        JobManager._print_logs_to_console(job_id, result_dir)

        # Route based on test_category
        if test_category == "accuracy":
            return JobManager._check_accuracy_result(
                job_id=job_id,
                metrics_config=metrics_config,
                accuracy_config=accuracy_config,
                result_dir=result_dir)
        else:  # perf
            return JobManager._check_perf_result(job_id=job_id,
                                                 benchmark_type=benchmark_type,
                                                 config=config,
                                                 metrics_config=metrics_config,
                                                 model_name=model_name,
                                                 result_dir=result_dir,
                                                 timestamps=timestamps,
                                                 test_name=test_name)


# create executor function
run_job = make_slurm_run_command()
