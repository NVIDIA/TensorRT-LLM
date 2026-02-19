"""Disaggregated Benchmark Test - YAML Configuration Based."""

import atexit

import pytest
from execution.executor import JobManager
from utils.common import CONFIG_BASE_DIR, EnvManager
from utils.config_loader import ConfigLoader, TestConfig
from utils.config_validator import ConfigValidator
from utils.logger import logger
from utils.trackers import TestCaseTracker, session_tracker

# Load all test configurations
config_loader = ConfigLoader(base_dir=CONFIG_BASE_DIR)
ALL_TEST_CONFIGS = config_loader.scan_configs()

# Separate performance, accuracy, and stress test configurations
PERF_TEST_CONFIGS = [c for c in ALL_TEST_CONFIGS if c.test_category == "perf"]
ACCURACY_TEST_CONFIGS = [c for c in ALL_TEST_CONFIGS if c.test_category == "accuracy"]
STRESS_TEST_CONFIGS = [c for c in ALL_TEST_CONFIGS if c.test_category == "stress"]

# Convert to pytest parameters
PERF_TEST_CASES = [pytest.param(config, id=config.test_id) for config in PERF_TEST_CONFIGS]
ACCURACY_TEST_CASES = [pytest.param(config, id=config.test_id) for config in ACCURACY_TEST_CONFIGS]
STRESS_TEST_CASES = [pytest.param(config, id=config.test_id) for config in STRESS_TEST_CONFIGS]

# Flag to track if session end has been called
_session_ended = False


def _ensure_session_end():
    """Ensure session end is called even on abnormal exit."""
    global _session_ended
    if not _session_ended:
        _session_ended = True
        logger.warning("Ensuring session cleanup...")
        session_tracker.end_and_collect()


# Register atexit handler
if not EnvManager.get_debug_mode():
    atexit.register(_ensure_session_end)
else:
    logger.debug(f"Debug mode: Skipping atexit handler: {EnvManager.get_debug_job_id()}")


@pytest.fixture(scope="session", autouse=True)
def session_lifecycle():
    """Session lifecycle management."""
    from utils.job_tracker import JobTracker

    # Record pytest main process PID for GitLab CI cleanup
    JobTracker.record_pid()

    session_tracker.start()
    try:
        yield
    finally:
        if not EnvManager.get_debug_mode():
            _ensure_session_end()
        else:
            logger.debug(f"Debug mode: Skipping session cleanup: {EnvManager.get_debug_job_id()}")


class TestDisaggBenchmark:
    """Disaggregated benchmark test class - YAML based."""

    @pytest.mark.perf
    @pytest.mark.parametrize("test_config", PERF_TEST_CASES)
    def test_benchmark(self, request, batch_manager, test_config: TestConfig):
        """Performance benchmark test for YAML configurations."""
        full_test_name = request.node.name

        # Note: Configuration validation is done during batch submission (in conftest.py)
        # If validation failed, job_id will be None and the assert below will fail

        # Create test case tracker
        test_tracker = TestCaseTracker()
        test_case_name = test_config.test_id

        # Start tracking test case
        test_tracker.start_test_case(test_case_name)

        job_id = None
        result = None

        try:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Performance Test: {test_config.display_name}")
            logger.info(f"Test ID: {test_config.test_id}")
            logger.info(f"Config file: {test_config.config_path}")
            logger.info(f"Test type: {test_config.test_type}")
            logger.info(f"Category: {test_config.test_category}")
            logger.info(f"Model: {test_config.model_name}")
            logger.info(f"Benchmark: {test_config.benchmark_type}")
            logger.info(f"Metrics log: {test_config.metrics_config.log_file}")
            logger.info(f"Supported GPUs: {', '.join(test_config.supported_gpus)}")
            logger.info(f"{'=' * 60}")

            if EnvManager.get_debug_mode():
                logger.debug(
                    f"Debug mode: Skipping job submission, using job_id: {EnvManager.get_debug_job_id()}"
                )
                job_id = EnvManager.get_debug_job_id()
            else:
                # Get job_id from batch manager (auto-submits batch if needed)
                job_id = batch_manager.get_job_id(test_config)

                # Validate submission result (will be None if validation/submission failed)
                error_msg = batch_manager.submit_errors.get(
                    test_config.test_id, "Check batch submission logs for details"
                )
                assert job_id, f"Failed to submit job for {test_config.test_id}\n{error_msg}"

                # Wait for completion (timeout: 15 hours = 54000 seconds)
                JobManager.wait_for_completion(job_id, 54000, test_config, check_early_failure=True)

            # End tracking test case
            test_tracker.end_test_case()

            # Get timestamps information
            timestamps = test_tracker.get_timestamps()

            # Check results and generate report
            result = JobManager.check_result(job_id, test_config, timestamps, full_test_name)
            assert result["success"], f"Performance test failed: {job_id}"

        except Exception as e:
            test_tracker.end_test_case()
            raise e
        finally:
            # Always backup logs, regardless of success or failure
            result_dir = JobManager.get_result_dir(test_config)
            is_passed = result.get("success", False) if result else False
            try:
                JobManager.backup_logs(job_id, test_config, result_dir, is_passed)
            except Exception as backup_error:
                logger.error(f"Failed to backup logs: {backup_error}")

    @pytest.mark.accuracy
    @pytest.mark.parametrize("test_config", ACCURACY_TEST_CASES)
    def test_accuracy(self, request, batch_manager, test_config: TestConfig):
        """Accuracy test for YAML configurations."""
        full_test_name = request.node.name

        # Validate configuration first (before any other operations)
        try:
            ConfigValidator.validate_test_config(test_config)
        except Exception as e:
            pytest.fail(f"Configuration validation failed: {e}")

        # Create test case tracker
        test_tracker = TestCaseTracker()
        test_case_name = test_config.test_id

        # Start tracking test case
        test_tracker.start_test_case(test_case_name)

        job_id = None
        result = None

        try:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Accuracy Test: {test_config.display_name}")
            logger.info(f"Test ID: {test_config.test_id}")
            logger.info(f"Config file: {test_config.config_path}")
            logger.info(f"Test type: {test_config.test_type}")
            logger.info(f"Model: {test_config.model_name}")

            # Log configured datasets
            if test_config.accuracy_config:
                dataset_names = test_config.accuracy_config.get_all_dataset_names()
                logger.info(f"Datasets: {', '.join(dataset_names)}")

            logger.info(f"Metrics log: {test_config.metrics_config.log_file}")
            logger.info(f"Supported GPUs: {', '.join(test_config.supported_gpus)}")
            logger.info(f"{'=' * 60}")

            if EnvManager.get_debug_mode():
                logger.debug(
                    f"Debug mode: Skipping job submission, using job_id: {EnvManager.get_debug_job_id()}"
                )
                job_id = EnvManager.get_debug_job_id()
            else:
                # Get job_id from batch manager (auto-submits batch if needed)
                job_id = batch_manager.get_job_id(test_config)

                # Validate submission result
                assert job_id, f"Failed to get job_id for {test_config.test_id}"

                # Wait for completion (timeout: 15 hours = 54000 seconds)
                JobManager.wait_for_completion(job_id, 54000, test_config, check_early_failure=True)

            # End tracking test case
            test_tracker.end_test_case()

            # Get timestamps information
            timestamps = test_tracker.get_timestamps()

            # Check results and validate accuracy
            result = JobManager.check_result(job_id, test_config, timestamps, full_test_name)
            assert result["success"], (
                f"Accuracy test failed: {result.get('error', 'Unknown error')}"
            )

        except Exception as e:
            test_tracker.end_test_case()
            raise e
        finally:
            # Always backup logs, regardless of success or failure
            result_dir = JobManager.get_result_dir(test_config)
            is_passed = result.get("success", False) if result else False
            try:
                JobManager.backup_logs(job_id, test_config, result_dir, is_passed)
            except Exception as backup_error:
                logger.error(f"Failed to backup logs: {backup_error}")

    @pytest.mark.stress
    @pytest.mark.parametrize("test_config", STRESS_TEST_CASES)
    def test_stress(self, request, batch_manager, test_config: TestConfig):
        """Stress test combining performance benchmarks and accuracy validation.

        This test type is designed for stress testing scenarios where both
        performance metrics (CSV output) and accuracy (e.g., GSM8K) need to be validated.
        """
        full_test_name = request.node.name

        # Note: Configuration validation is done during batch submission (in conftest.py)
        # If validation failed, job_id will be None and the assert below will fail

        # Create test case tracker
        test_tracker = TestCaseTracker()
        test_case_name = test_config.test_id

        # Start tracking test case
        test_tracker.start_test_case(test_case_name)

        job_id = None
        result = None

        try:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Stress Test (Perf + Accuracy): {test_config.display_name}")
            logger.info(f"Test ID: {test_config.test_id}")
            logger.info(f"Config file: {test_config.config_path}")
            logger.info(f"Test type: {test_config.test_type}")
            logger.info(f"Category: {test_config.test_category}")
            logger.info(f"Model: {test_config.model_name}")
            logger.info(f"Benchmark: {test_config.benchmark_type}")

            # Log accuracy datasets if configured
            if test_config.accuracy_config:
                dataset_names = test_config.accuracy_config.get_all_dataset_names()
                logger.info(f"Accuracy Datasets: {', '.join(dataset_names)}")

            logger.info(f"Metrics log: {test_config.metrics_config.log_file}")
            logger.info(f"Supported GPUs: {', '.join(test_config.supported_gpus)}")
            logger.info(f"{'=' * 60}")

            if EnvManager.get_debug_mode():
                logger.debug(
                    f"Debug mode: Skipping job submission, using job_id: {EnvManager.get_debug_job_id()}"
                )
                job_id = EnvManager.get_debug_job_id()
            else:
                # Get job_id from batch manager (auto-submits batch if needed)
                job_id = batch_manager.get_job_id(test_config)

                # Validate submission result (will be None if validation/submission failed)
                error_msg = batch_manager.submit_errors.get(
                    test_config.test_id, "Check batch submission logs for details"
                )
                assert job_id, f"Failed to submit job for {test_config.test_id}\n{error_msg}"

                # Wait for completion (timeout: 15 hours = 54000 seconds)
                JobManager.wait_for_completion(job_id, 54000, test_config, check_early_failure=True)

            # End tracking test case
            test_tracker.end_test_case()

            # Get timestamps information
            timestamps = test_tracker.get_timestamps()

            # Check results - this will handle both perf CSV writing AND accuracy validation
            result = JobManager.check_result(job_id, test_config, timestamps, full_test_name)
            assert result["success"], f"Stress test failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            test_tracker.end_test_case()
            raise e
        finally:
            # Always backup logs, regardless of success or failure
            result_dir = JobManager.get_result_dir(test_config)
            is_passed = result.get("success", False) if result else False
            try:
                JobManager.backup_logs(job_id, test_config, result_dir, is_passed)
            except Exception as backup_error:
                logger.error(f"Failed to backup logs: {backup_error}")


if __name__ == "__main__":
    """Run benchmark tests"""
    pytest.main([__file__, "-v"])
