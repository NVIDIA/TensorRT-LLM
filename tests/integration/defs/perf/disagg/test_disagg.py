"""Disaggregated Benchmark Test - YAML Configuration Based."""

import atexit

import pytest
from common import CONFIG_BASE_DIR, DEBUG_JOB_ID, DEBUG_MODE
from config_loader import ConfigLoader, TestConfig
from executor import JobManager
from trackers import TestCaseTracker, session_tracker

# Load all test configurations
config_loader = ConfigLoader(base_dir=CONFIG_BASE_DIR)
ALL_TEST_CONFIGS = config_loader.scan_configs()

# Separate performance and accuracy test configurations
PERF_TEST_CONFIGS = [c for c in ALL_TEST_CONFIGS if c.test_category == "perf"]
ACCURACY_TEST_CONFIGS = [
    c for c in ALL_TEST_CONFIGS if c.test_category == "accuracy"
]

# Convert to pytest parameters
PERF_TEST_CASES = [
    pytest.param(config, id=config.test_id) for config in PERF_TEST_CONFIGS
]
ACCURACY_TEST_CASES = [
    pytest.param(config, id=config.test_id) for config in ACCURACY_TEST_CONFIGS
]

# Flag to track if session end has been called
_session_ended = False


def _ensure_session_end():
    """Ensure session end is called even on abnormal exit."""
    global _session_ended
    if not _session_ended:
        _session_ended = True
        print("\n⚠️  Ensuring session cleanup...")
        session_tracker.end_and_collect()


# Register atexit handler
if not DEBUG_MODE:
    atexit.register(_ensure_session_end)
else:
    print(f"🐛 Debug mode: Skipping atexit handler: {DEBUG_JOB_ID}")


@pytest.fixture(scope="session", autouse=True)
def session_lifecycle():
    """Session lifecycle management."""
    session_tracker.start()
    try:
        yield
    finally:
        if not DEBUG_MODE:
            _ensure_session_end()
        else:
            print(f"🐛 Debug mode: Skipping session cleanup: {DEBUG_JOB_ID}")


class TestDisaggBenchmark:
    """Disaggregated benchmark test class - YAML based."""

    @pytest.mark.perf
    @pytest.mark.parametrize("test_config", PERF_TEST_CASES)
    def test_benchmark(self, request, test_config: TestConfig):
        """Performance benchmark test for YAML configurations."""
        full_test_name = request.node.name

        # Create test case tracker
        test_tracker = TestCaseTracker()
        test_case_name = f"{test_config.model_name}-{test_config.benchmark_type}"

        # Start tracking test case
        test_tracker.start_test_case(test_case_name)

        try:
            print(f"\n{'=' * 60}")
            print(f"Performance Test: {test_config.display_name}")
            print(f"Test ID: {test_config.test_id}")
            print(f"Config file: {test_config.config_path}")
            print(f"Test type: {test_config.test_type}")
            print(f"Category: {test_config.test_category}")
            print(f"Model: {test_config.model_name}")
            print(f"Benchmark: {test_config.benchmark_type}")
            print(f"Metrics log: {test_config.metrics_config.log_file}")
            print(f"Supported GPUs: {', '.join(test_config.supported_gpus)}")
            print(f"{'=' * 60}")
            if DEBUG_MODE:
                print(
                    f"🐛 Debug mode: Skipping job submission, using job_id: {DEBUG_JOB_ID}"
                )
                job_id = DEBUG_JOB_ID
            else:
                # Submit job using JobManager
                success, job_id = JobManager.submit_job(test_config)

                # Validate submission result
                assert success, f"Job submission failed: {test_config.test_id}"
                assert job_id, "Unable to get job ID"

                # Wait for completion (with early failure detection)
                completed, error_msg = JobManager.wait_for_completion(
                    job_id, 7200, test_config, check_early_failure=True)
                if not completed:
                    JobManager.cancel_job(job_id)
                    result_dir = JobManager.get_result_dir(test_config)
                    JobManager.backup_logs(job_id, test_config, result_dir,
                                           False)
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
            result = JobManager.check_result(job_id, test_config, timestamps,
                                             full_test_name)
            assert result["success"], f"Performance test failed: {job_id}"

        except Exception as e:
            test_tracker.end_test_case()
            raise e

    @pytest.mark.accuracy
    @pytest.mark.parametrize("test_config", ACCURACY_TEST_CASES)
    def test_accuracy(self, request, test_config: TestConfig):
        """Accuracy test for YAML configurations."""
        full_test_name = request.node.name

        # Create test case tracker
        test_tracker = TestCaseTracker()
        test_case_name = f"{test_config.model_name}-accuracy"

        # Start tracking test case
        test_tracker.start_test_case(test_case_name)

        try:
            print(f"\n{'=' * 60}")
            print(f"Accuracy Test: {test_config.display_name}")
            print(f"Test ID: {test_config.test_id}")
            print(f"Config file: {test_config.config_path}")
            print(f"Test type: {test_config.test_type}")
            print(f"Model: {test_config.model_name}")

            # Print configured datasets
            if test_config.accuracy_config:
                dataset_names = test_config.accuracy_config.get_all_dataset_names(
                )
                print(f"Datasets: {', '.join(dataset_names)}")

            print(f"Metrics log: {test_config.metrics_config.log_file}")
            print(f"Supported GPUs: {', '.join(test_config.supported_gpus)}")
            print(f"{'=' * 60}")

            if DEBUG_MODE:
                print(
                    f"🐛 Debug mode: Skipping job submission, using job_id: {DEBUG_JOB_ID}"
                )
                job_id = DEBUG_JOB_ID
            else:
                # Submit job using JobManager
                success, job_id = JobManager.submit_job(test_config)

                # Validate submission result
                assert success, f"Job submission failed: {test_config.test_id}"
                assert job_id, "Unable to get job ID"

                # Wait for completion (accuracy tests may need more time - 3 hours timeout)
                completed, error_msg = JobManager.wait_for_completion(
                    job_id, 7200, test_config, check_early_failure=True)
                if not completed:
                    JobManager.cancel_job(job_id)
                    result_dir = JobManager.get_result_dir(test_config)
                    JobManager.backup_logs(job_id, test_config, result_dir,
                                           False)
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
            result = JobManager.check_result(job_id, test_config, timestamps,
                                             full_test_name)
            assert result[
                "success"], f"Accuracy test failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            test_tracker.end_test_case()
            raise e


if __name__ == "__main__":
    """Run benchmark tests"""
    pytest.main([__file__, "-v"])
