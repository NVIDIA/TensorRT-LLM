"""Disaggregated Benchmark Executor."""

import os
import re
import shutil
import time
from typing import Any, Dict, Optional

import yaml
from reporting.report import LogParser, LogWriter, ResultSaver
from utils.common import EnvManager
from utils.logger import logger

from execution.subprocess_utils import exec_cmd, exec_cmd_with_output

# ============================================================================
# Job Manager
# ============================================================================


class JobManager:
    """Job manager class for test jobs and session collection."""

    # ============================================================================
    # Generic Job Submission (Direct sbatch)
    # ============================================================================

    @staticmethod
    def submit_shell_job(
        job_name: str,
        script_path: str,
        script_args: list[str] = None,
        output_log_file: str = None,
        timeout: int = 7200,
        container_name: str = None,
    ) -> tuple[bool, str]:
        """Submit a generic shell script job using sbatch --wrap.

        This is a low-level method for submitting shell scripts to SLURM
        via sbatch --wrap (non-blocking). Supports executing script files
        with arguments inside containers.

        Args:
            job_name: SLURM job name
            script_path: Path to the shell script file to execute
            script_args: List of arguments to pass to the script (optional)
            output_log_file: Full path to output log file (optional, defaults to OUTPUT_PATH/{job_name}.log)
            timeout: Job timeout in seconds (default: 7200 = 2 hours)
            container_name: Container name for srun (optional, defaults to job_name)

        Returns:
            tuple: (success: bool, job_id: str)
        """
        try:
            # Get environment configuration
            container_image = EnvManager.get_container_image()
            container_mount = EnvManager.get_container_mount()
            output_path = EnvManager.get_output_path()

            # Ensure output directory exists
            os.makedirs(output_path, exist_ok=True)

            # Set defaults
            if output_log_file is None:
                output_log_file = f"{output_path}/{job_name}.log"
            if container_name is None:
                container_name = job_name
            if script_args is None:
                script_args = []

            # Build the bash command with script and arguments
            # Quote the script path and each argument separately
            quoted_script = f'"{script_path}"'
            quoted_args = " ".join(f'"{arg}"' for arg in script_args)
            bash_command = f"bash {quoted_script} {quoted_args}".strip()

            # Build complete srun command (runs inside sbatch)
            srun_command = (
                f"srun -l "
                f"--container-name={container_name} "
                f"--container-image={container_image} "
                f"--container-mounts={container_mount} "
                f"{bash_command}"
            )

            # Convert timeout to HH:MM:SS format
            hours = timeout // 3600
            minutes = (timeout % 3600) // 60
            seconds = timeout % 60
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            sbatch_args = [
                "sbatch",
                f"--job-name={job_name}",
                f"--partition={EnvManager.get_slurm_partition()}",
                f"--account={EnvManager.get_slurm_account()}",
                f"--time={time_str}",
                "--nodes=1",
                "--ntasks=1",
                f"--output={output_log_file}",
                "--parsable",  # Easier job ID parsing
            ]

            # Add extra SLURM arguments (including --gres from GPU_RESOURCE_CONFIG)
            slurm_extra_args = EnvManager.get_slurm_extra_args()
            if slurm_extra_args:
                sbatch_args.append(slurm_extra_args)

            # Add --wrap with the srun command
            sbatch_args.extend(["--wrap", srun_command])

            # Submit the job
            logger.info(f"Submitting job '{job_name}' (using sbatch --wrap)...")
            logger.debug(f"Script: {script_path}")
            logger.debug(f"Log file: {output_log_file}")

            # Use check=False to allow submission even with Kerberos warnings
            # (mimics submit.py behavior)
            output = exec_cmd_with_output(sbatch_args, timeout=60, check=False)
            job_id = output.strip()

            # Parse job ID (--parsable returns just the job ID)
            if job_id.isdigit():
                logger.success(f"Job '{job_name}' submitted: {job_id}")
                logger.info(f"All logs will be written to: {output_log_file}")
                return True, job_id

            # Fallback: try to extract from "Submitted batch job" format
            match = re.search(r"Submitted batch job (\d+)", output)
            if match:
                job_id = match.group(1)
                logger.success(f"Job '{job_name}' submitted: {job_id}")
                return True, job_id

            logger.error(f"Failed to parse job ID from output: {output}")
            return False, ""

        except Exception as e:
            logger.error(f"Failed to submit job '{job_name}': {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return False, str(e)

    # ============================================================================
    # Session Collection Job Submission
    # ============================================================================

    @staticmethod
    def submit_session_collect_job() -> tuple[bool, str]:
        """Submit session collect job using sbatch (non-blocking).

        This method prepares the arguments for the session_collect.sh script
        and submits it via the generic submit_shell_job() method.

        Key benefits:
        - Non-blocking execution (pytest doesn't wait)
        - Better resource scheduling (queues if resources unavailable)
        - Fault tolerance (job survives parent process exit)
        - Unified job management (reuses wait_for_completion)
        - All logs redirected to session_collect.log

        Returns:
            tuple: (success: bool, job_id: str)
        """
        try:
            # Get environment configuration
            work_dir = EnvManager.get_work_dir()
            repo_dir = EnvManager.get_repo_dir()
            install_mode = EnvManager.get_install_mode()
            trtllm_wheel_path = EnvManager.get_trtllm_wheel_path()
            output_path = EnvManager.get_output_path()

            # Prepare script path and arguments
            script_path = f"{work_dir}/session_collect.sh"
            script_args = [install_mode, repo_dir, work_dir, output_path, trtllm_wheel_path]

            # Submit using the generic shell job method
            return JobManager.submit_shell_job(
                job_name="session_collect",
                script_path=script_path,
                script_args=script_args,
                output_log_file=f"{output_path}/session_collect.log",
                timeout=7200,  # 2 hours
                container_name="session-collect",
            )

        except Exception as e:
            logger.error(f"Failed to prepare session collect job: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return False, str(e)

    # ============================================================================
    # Test Job Submission (Via submit.py script)
    # ============================================================================

    @staticmethod
    def submit_test_job(test_config) -> tuple:
        """Submit benchmark test job using submit.py script.

        This method submits test jobs by calling the submit.py script,
        which handles test-specific configuration and SLURM job setup.

        Args:
            test_config: TestConfig object containing configuration

        Returns:
            tuple: (success: bool, job_id: str)
        """
        logger.info("Submitting test job via submit.py...")

        try:
            import re

            # Get pre-calculated temporary config file path from test_config
            temp_config_path = test_config.temp_config_path

            # Write temporary config file with replaced environment variables
            logger.info(f"Creating temporary config: {temp_config_path}")
            with open(temp_config_path, "w") as f:
                yaml.dump(
                    test_config.config_data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    width=1000,
                )
            logger.success(f"Temporary config created: {os.path.basename(temp_config_path)}")

            # Call submit.py with the temporary config file
            submit_script = os.path.join(EnvManager.get_script_dir(), "submit.py")

            case_log_dir = JobManager.get_result_dir(test_config)

            cmd = ["python3", submit_script, "-c", temp_config_path, "--log-dir", case_log_dir]

            logger.info(f"Command: {' '.join(cmd)}")

            # Execute submission
            output = exec_cmd_with_output(cmd, timeout=60)
            logger.info(f"Output: {output}")

            # Parse job ID from output
            if "Submitted batch job" in output:
                match = re.search(r"Submitted batch job (\d+)", output)
                if match:
                    job_id = match.group(1)
                    logger.success(f"Job submitted successfully: {job_id}")
                    return True, job_id

            logger.error("Unable to extract job ID from output")
            # Clean up temporary file if submission failed
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            return False, ""

        except Exception as e:
            error_msg = str(e)
            # Extract stderr from CalledProcessError if available
            if hasattr(e, "stderr") and e.stderr:
                error_msg = e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr
            logger.error(f"Job submission exception: {error_msg}")
            # Clean up temporary file on exception
            temp_config_path = test_config.temp_config_path
            # if os.path.exists(temp_config_path):
            #     os.remove(temp_config_path)
            return False, error_msg

    @staticmethod
    def backup_logs(
        job_id: Optional[str],
        test_config,
        result_dir: str,
        is_passed: bool,
    ) -> Optional[str]:
        """Backup logs and config files to test_id directory.

        Args:
            job_id: SLURM job ID (None if submission failed)
            test_config: TestConfig object
            result_dir: Result directory path (already named as test_id)
            is_passed: Whether the job passed
        Returns:
            Final directory path if successful, None otherwise
        """
        if job_id is None:
            logger.warning(f"Job submission failed for {test_config.test_id}")
        else:
            logger.info(f"Backing up logs for job {job_id} ({test_config.test_id})")

        if not os.path.exists(result_dir):
            logger.warning(f"Result directory does not exist yet: {result_dir}")
            return None

        try:
            final_dir = result_dir

            # For FAILED cases, rename directory to add _ERROR suffix
            if not is_passed:
                error_dir = f"{result_dir}_ERROR"
                logger.info(f"Renaming failed case directory: {result_dir} -> {error_dir}")

                # Remove old error directory if exists
                if os.path.exists(error_dir):
                    logger.warning(f"Removing existing error directory: {error_dir}")
                    shutil.rmtree(error_dir)

                # Rename to add _ERROR suffix
                shutil.move(result_dir, error_dir)
                final_dir = error_dir
                logger.success(f"Directory renamed to: {final_dir}")

            # Copy temporary config file to the directory
            temp_config_path = test_config.temp_config_path
            if os.path.exists(temp_config_path):
                dest_path = os.path.join(final_dir, os.path.basename(temp_config_path))
                shutil.copy(temp_config_path, dest_path)
                logger.success(f"Temporary config copied to: {dest_path}")
                # Clean up the original temp config file
                os.remove(temp_config_path)
                logger.info(f"Cleaned up temporary config: {temp_config_path}")
            else:
                logger.warning(f"Temporary config not found: {temp_config_path}")

            return final_dir

        except Exception as e:
            logger.warning(f"Failed to backup logs: {e}")
            # Try to clean up temporary file on backup failure
            temp_config_path = test_config.temp_config_path
            if os.path.exists(temp_config_path):
                try:
                    os.remove(temp_config_path)
                    logger.info(f"Cleaned up temp config after backup failure: {temp_config_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp config: {cleanup_error}")
            return None

    @staticmethod
    def get_result_dir(test_config) -> str:
        """Get result directory.

        Args:
            test_config: TestConfig object

        Returns:
            Result directory path
        """
        # Use the same path as in submit_job: {output_path}/slurm_logs/{test_id}
        log_dir = os.path.join(EnvManager.get_output_path(), "slurm_logs")
        case_log_dir = os.path.join(log_dir, test_config.test_id.replace(":", "-"))
        return case_log_dir

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

        Note: backup_logs should be called separately by the caller (test_disagg.py).

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
        logger.info(f"Result directory: {result_dir}")

        # Initialize default result in case of exception
        check_result = {"job_id": job_id, "status": "ERROR", "success": False}

        try:
            # Call the internal implementation method
            check_result = JobManager._check_job_result(
                job_id=job_id,
                test_category=test_config.test_category,  # Pass test category for routing
                benchmark_type=test_config.benchmark_type,
                config=config_data,
                metrics_config=test_config.metrics_config,
                accuracy_config=test_config.accuracy_config,  # Pass accuracy config
                model_name=test_config.model_name,
                result_dir=result_dir,
                timestamps=timestamps,
                test_name=test_name,
            )
        except Exception as e:
            logger.error(f"Exception during result checking: {e}")
            check_result["error"] = f"Exception during result checking: {str(e)}"
        return check_result

    @staticmethod
    def check_for_early_failure(job_id: str, test_config) -> tuple[bool, Optional[str]]:
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
                        if (is_gen_log or is_ctx_log) and filename.endswith(".log"):
                            log_path = os.path.join(result_dir, filename)
                            try:
                                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                                    # Read last 100KB
                                    f.seek(0, 2)
                                    file_size = f.tell()
                                    f.seek(max(0, file_size - 102400), 0)
                                    recent_content = f.read()

                                    for pattern, error_msg in error_patterns:
                                        if re.search(pattern, recent_content, re.MULTILINE):
                                            return True, f"{error_msg} in {filename}"
                            except Exception as e:
                                logger.warning(f"Failed to check {filename}: {e}")
            except Exception:
                # result_dir might not exist yet, that's OK
                pass

        except Exception as e:
            logger.warning(f"Error during early failure check: {e}")

        return False, None

    @staticmethod
    def check_job_exists(job_id: str) -> bool:
        """Check if job still exists in SLURM queue.

        Returns:
            True if job exists (running or pending), False if job is gone
        """
        try:
            # Use squeue to check if job exists in queue
            squeue_output = exec_cmd_with_output(["squeue", "-j", job_id, "--noheader"], timeout=30)
            # If output is not empty, job exists
            return bool(squeue_output.strip())
        except Exception as e:
            # If command fails, assume job doesn't exist
            logger.debug(f"squeue check failed (job likely finished): {e}")
            return False

    @staticmethod
    def wait_for_completion(
        job_id: str, timeout: int = 3600, test_config=None, check_early_failure: bool = True
    ) -> None:
        """Wait for job to finish (disappear from queue).

        Simplified logic: Just wait until job no longer exists in SLURM queue,
        regardless of final status (COMPLETED, CANCELLED, FAILED, etc).
        If timeout or early failure detected, cancel the job.
        The actual success/failure will be determined by log file parsing.

        Args:
            job_id: SLURM job ID
            timeout: Maximum wait time in seconds
            test_config: TestConfig object (required for early failure detection)
            check_early_failure: Whether to check logs for early failures
        """
        start_time = time.time()
        check_interval = 180  # Check every 3 minutes
        failure_check_interval = 60  # Check for failures every 60 seconds
        last_failure_check = start_time

        # Wait for job to appear in system (initial delay)
        logger.info(f"Waiting for job {job_id} to start...")
        time.sleep(60)  # Initial wait for job to be scheduled

        logger.info(f"Waiting for job {job_id} to finish...")

        while time.time() - start_time < timeout:
            # Simple check: does job still exist?
            job_exists = JobManager.check_job_exists(job_id)

            if not job_exists:
                # Job has disappeared from queue - it's done (whatever the status was)
                logger.success(f"Job {job_id} finished (no longer in queue)")
                return

            # Check for early failures (only if test_config is provided)
            current_time = time.time()
            if (
                check_early_failure
                and test_config
                and current_time - last_failure_check >= failure_check_interval
            ):
                has_error, error_msg = JobManager.check_for_early_failure(job_id, test_config)
                if has_error:
                    logger.error(f"Early failure detected: {error_msg}")
                    logger.warning(f"Cancelling job {job_id} due to early failure")
                    JobManager.cancel_job(job_id)
                    # Wait a bit for job to be cancelled, then return
                    time.sleep(10)
                    return
                last_failure_check = current_time

            time.sleep(check_interval)

        # Timeout - cancel the job
        logger.warning(f"Job {job_id} timeout after {timeout} seconds, cancelling...")
        JobManager.cancel_job(job_id)
        # Wait a bit for job to be cancelled
        time.sleep(10)

    @staticmethod
    def cancel_job(job_id: str) -> bool:
        """Cancel job."""
        try:
            exec_cmd(["scancel", job_id], timeout=30)
            logger.warning(f"Job cancelled: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Job cancellation failed: {e}")
            return False

    @staticmethod
    def _print_logs_to_console(job_id: str, result_dir: str) -> None:
        """Print SLURM log and all .log/.yaml files in result_dir to console.

        Args:
            job_id: SLURM job ID for finding the slurm log file
            result_dir: Result directory containing log and config files
        """
        # Print the slurm log to console (check if exists first)
        slurm_log_path = os.path.join(EnvManager.get_work_dir(), f"slurm-{job_id}.out")
        if os.path.exists(slurm_log_path):
            slurm_log_writer = LogWriter(EnvManager.get_work_dir())
            slurm_log_writer.print_to_console(f"slurm-{job_id}.out")
        else:
            logger.warning(f"SLURM log file not found: {slurm_log_path}")

        # Print all .log and .yaml files in result_dir (except output_server.log)
        if not os.path.exists(result_dir):
            logger.warning(f"Result directory not found: {result_dir}")
            return

        log_writer = LogWriter(result_dir)
        files_to_print = []
        for file in os.listdir(result_dir):
            if (file.endswith(".log") or file.endswith(".yaml")) and file != "output_server.log":
                files_to_print.append(file)

        # Sort files for consistent output order
        files_to_print.sort()

        for file in files_to_print:
            file_path = os.path.join(result_dir, file)
            if os.path.exists(file_path):
                log_writer.print_to_console(file)
            else:
                logger.warning(f"Log file not found: {file}")

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
        result: Dict[str, Any] = {"job_id": job_id, "status": "UNKNOWN", "success": False}

        # Validate accuracy_config
        if not accuracy_config:
            result["error"] = "Accuracy config not found in test configuration"
            return result

        # Check if result_dir exists
        if not os.path.exists(result_dir):
            error_msg = f"Result directory not found: {result_dir}"
            logger.error(error_msg)
            result["error"] = error_msg
            return result

        # Import and use AccuracyParser
        # Note: AccuracyParser handles log file checking with glob pattern support
        from reporting.accuracy_parser import AccuracyParser

        accuracy_parser = AccuracyParser(metrics_config, accuracy_config, result_dir)
        validation_result = accuracy_parser.parse_and_validate()

        # Check if parsing succeeded
        if not validation_result["success"]:
            result["error"] = validation_result.get("error", "Accuracy validation failed")
            return result

        # Log validation results
        logger.info("Accuracy Validation Results:")
        all_passed = validation_result["all_passed"]

        # Log results for each run (using dataclass attributes for type safety)
        for run_validation in validation_result.get("runs", []):
            run_name = run_validation.run_name
            run_passed = run_validation.all_passed
            status = "PASSED" if run_passed else "FAILED"

            logger.info(f"[{status}] {run_name}:")

            for ds_result in run_validation.results:
                status = "PASSED" if ds_result.passed else "FAILED"
                dataset_name = ds_result.dataset
                filter_type = ds_result.filter
                threshold_type = ds_result.threshold_type

                logger.info(f"   [{status}] {dataset_name} ({filter_type}) - {threshold_type}:")
                if ds_result.error:
                    logger.error(f"      Error: {ds_result.error}")
                else:
                    logger.info(f"      Expected: {ds_result.expected:.4f}")
                    logger.info(f"      Actual:   {ds_result.actual:.4f}")
                    logger.info(f"      Threshold type:  {ds_result.threshold_type}")
                    logger.info(f"      {ds_result.message}")

        # Set result status
        if all_passed:
            logger.success("All accuracy tests PASSED (all runs)")
            result["success"] = True
            result["status"] = "PASSED"
        else:
            logger.failure("Some accuracy tests FAILED")
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

        # Check if result_dir exists
        if not os.path.exists(result_dir):
            error_msg = f"Result directory not found: {result_dir}"
            logger.error(error_msg)
            result["error"] = error_msg
            return result

        # Check if required log file exists (6_bench.log)
        bench_log = os.path.join(result_dir, "6_bench.log")
        if not os.path.exists(bench_log):
            error_msg = f"Benchmark log file not found: {bench_log}"
            logger.error(error_msg)
            result["error"] = error_msg
            return result

        # Parse metrics and save to CSV
        log_parser = LogParser(benchmark_type, config, metrics_config, result_dir)
        parse_result = log_parser.parse(model_name, timestamps=timestamps, test_name=test_name)

        if not parse_result["status"]:
            result["error"] = "Failed to parse benchmark logs"
            return result

        # Check if df is None
        result_df = parse_result.get("df")
        if result_df is None:
            logger.error("Parse result contains None DataFrame")
            result["error"] = "Parse result contains None DataFrame"
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
            test_category: Test category ("perf", "accuracy", or "stress")
            benchmark_type: Benchmark type (1k1k, 8k1k, etc.)
            config: Configuration dict (YAML data)
            metrics_config: MetricsConfig object (default or custom)
            accuracy_config: AccuracyConfig object (required for accuracy and stress tests)
            model_name: Model name
            result_dir: Result directory
            timestamps: Optional timestamps dict
            test_name: Optional test name

        Returns:
            Dict with success status and details
            For stress tests, includes both perf and accuracy results
        """
        logger.info(f"Checking result directory: {result_dir}")

        # Print logs and config files to console
        JobManager._print_logs_to_console(job_id, result_dir)

        # Route based on test_category
        if test_category == "accuracy":
            # Use metrics config from accuracy_config (defaults to _COMMON_ACCURACY_METRICS)
            accuracy_metrics = accuracy_config.get_metrics_config()
            return JobManager._check_accuracy_result(
                job_id=job_id,
                metrics_config=accuracy_metrics,
                accuracy_config=accuracy_config,
                result_dir=result_dir,
            )
        elif test_category == "stress":
            # Stress tests combine both perf and accuracy validation
            # First check performance and write CSV
            perf_result = JobManager._check_perf_result(
                job_id=job_id,
                benchmark_type=benchmark_type,
                config=config,
                metrics_config=metrics_config,
                model_name=model_name,
                result_dir=result_dir,
                timestamps=timestamps,
                test_name=test_name,
            )

            # If perf check failed, return immediately
            if not perf_result.get("success", False):
                return perf_result

            # Then check accuracy if accuracy_config is provided
            if accuracy_config:
                # Use metrics config from accuracy_config (defaults to _COMMON_ACCURACY_METRICS)
                accuracy_metrics = accuracy_config.get_metrics_config()

                accuracy_result = JobManager._check_accuracy_result(
                    job_id=job_id,
                    metrics_config=accuracy_metrics,
                    accuracy_config=accuracy_config,
                    result_dir=result_dir,
                )

                # If accuracy check failed, merge results and return
                if not accuracy_result.get("success", False):
                    return {
                        **perf_result,
                        "success": False,
                        "accuracy_result": accuracy_result,
                        "error": f"Perf passed but accuracy failed: {accuracy_result.get('error', 'Unknown')}",
                    }

                # Both passed, merge results
                return {
                    **perf_result,
                    "accuracy_result": accuracy_result,
                    "success": True,
                }
            else:
                # No accuracy config, just return perf result
                logger.warning("Stress test has no accuracy_config, only perf validation performed")
                return perf_result
        else:  # perf
            return JobManager._check_perf_result(
                job_id=job_id,
                benchmark_type=benchmark_type,
                config=config,
                metrics_config=metrics_config,
                model_name=model_name,
                result_dir=result_dir,
                timestamps=timestamps,
                test_name=test_name,
            )
