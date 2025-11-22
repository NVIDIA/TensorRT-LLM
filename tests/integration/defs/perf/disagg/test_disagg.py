"""Disaggregated Benchmark Test - YAML Configuration Based."""

import atexit

import pytest
from utils.common import CONFIG_BASE_DIR, EnvManager
from utils.config_loader import ConfigLoader, TestConfig
from utils.config_validator import ConfigValidator
from utils.logger import logger
from utils.trackers import TestCaseTracker, session_tracker
from execution.executor import JobManager

# Load all test configurations
config_loader = ConfigLoader(base_dir=CONFIG_BASE_DIR)
ALL_TEST_CONFIGS = config_loader.scan_configs()

# Separate performance and accuracy test configurations
PERF_TEST_CONFIGS = [c for c in ALL_TEST_CONFIGS if c.test_category == "perf"]
ACCURACY_TEST_CONFIGS = [c for c in ALL_TEST_CONFIGS if c.test_category == "accuracy"]

# Convert to pytest parameters
PERF_TEST_CASES = [pytest.param(config, id=config.test_id) for config in PERF_TEST_CONFIGS]
ACCURACY_TEST_CASES = [pytest.param(config, id=config.test_id) for config in ACCURACY_TEST_CONFIGS]

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
    def test_benchmark(self, request, test_config: TestConfig):
        """Performance benchmark test for YAML configurations."""
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
                logger.debug(f"Debug mode: Skipping job submission, using job_id: {EnvManager.get_debug_job_id()}")
                job_id = EnvManager.get_debug_job_id()
            else:
                # Submit job using JobManager
                success, job_id = JobManager.submit_job(test_config)

                # Validate submission result
                assert success, f"Job submission failed: {test_config.test_id}"
                assert job_id, "Unable to get job ID"

                # Wait for completion (with early failure detection)
                completed, error_msg = JobManager.wait_for_completion(
                    job_id, 7200, test_config, check_early_failure=True
                )
                if not completed:
                    JobManager.cancel_job(job_id)
                    result_dir = JobManager.get_result_dir(test_config)
                    JobManager.backup_logs(job_id, test_config, result_dir, False)
                    JobManager.cleanup_result_dir(result_dir)
                    # Provide detailed error message
                    if error_msg == "timeout":
                        assert False, f"Job execution timeout after 7200s: {job_id}"
                    else:
                        assert False, f"Job failed early: {error_msg} (job_id: {job_id})"

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

    @pytest.mark.accuracy
    @pytest.mark.parametrize("test_config", ACCURACY_TEST_CASES)
    def test_accuracy(self, request, test_config: TestConfig):
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
                logger.debug(f"Debug mode: Skipping job submission, using job_id: {EnvManager.get_debug_job_id()}")
                job_id = EnvManager.get_debug_job_id()
            else:
                # Submit job using JobManager
                success, job_id = JobManager.submit_job(test_config)

                # Validate submission result
                assert success, f"Job submission failed: {test_config.test_id}"
                assert job_id, "Unable to get job ID"

                # Wait for completion (accuracy tests may need more time - 3 hours timeout)
                completed, error_msg = JobManager.wait_for_completion(
                    job_id, 7200, test_config, check_early_failure=True
                )
                if not completed:
                    JobManager.cancel_job(job_id)
                    result_dir = JobManager.get_result_dir(test_config)
                    JobManager.backup_logs(job_id, test_config, result_dir, False)
                    JobManager.cleanup_result_dir(result_dir)
                    # Provide detailed error message
                    if error_msg == "timeout":
                        assert False, f"Accuracy test timeout after 10800s: {job_id}"
                    else:
                        assert False, f"Accuracy test failed early: {error_msg} (job_id: {job_id})"

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


if __name__ == "__main__":
    """Run benchmark tests"""
    pytest.main([__file__, "-v"])
