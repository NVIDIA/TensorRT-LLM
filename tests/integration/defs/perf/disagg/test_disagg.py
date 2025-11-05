"""Disaggregated Benchmark Test - YAML Configuration Based."""

import atexit

import pytest
from common import CONFIG_BASE_DIR
from config_loader import ConfigLoader, TestConfig
from executor import JobManager
from trackers import TestCaseTracker, session_tracker

# Load all test configurations
config_loader = ConfigLoader(base_dir=CONFIG_BASE_DIR)
ALL_TEST_CONFIGS = config_loader.scan_configs()

# Convert to pytest parameters
ALL_TEST_CASES = [pytest.param(config, id=config.test_id) for config in ALL_TEST_CONFIGS]


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
atexit.register(_ensure_session_end)


@pytest.fixture(scope="session", autouse=True)
def session_lifecycle():
    """Session lifecycle management."""
    session_tracker.start()
    try:
        yield
    finally:
        _ensure_session_end()


class TestDisaggBenchmark:
    """Disaggregated benchmark test class - YAML based."""

    @pytest.mark.parametrize("test_config", ALL_TEST_CASES)
    def test_benchmark(self, request, test_config: TestConfig):
        """Benchmark test for YAML configurations."""
        full_test_name = request.node.name

        # Create test case tracker
        test_tracker = TestCaseTracker()
        test_case_name = f"{test_config.model_name}-{test_config.benchmark_type}"

        # Start tracking test case
        test_tracker.start_test_case(test_case_name)

        try:
            print(f"\n{'=' * 60}")
            print(f"Test: {test_config.display_name}")
            print(f"Config file: {test_config.config_path}")
            print(f"Test type: {test_config.test_type}")
            print(f"Category: {test_config.test_category}")
            print(f"Model: {test_config.model_name}")
            print(f"Benchmark: {test_config.benchmark_type}")
            print(f"Metrics log: {test_config.metrics_config.log_file}")
            print(f"Supported GPUs: {', '.join(test_config.supported_gpus)}")
            print(f"{'=' * 60}")

            # Submit job using JobManager
            success, job_id = JobManager.submit_job(test_config)

            # Validate submission result
            assert success, f"Job submission failed: {test_config.test_id}"
            assert job_id, "Unable to get job ID"

            # Wait for completion
            completed = JobManager.wait_for_completion(job_id, 7200)
            if not completed:
                JobManager.cancel_job(job_id)
                assert False, f"Job execution timeout: {job_id}"

            # End tracking test case
            test_tracker.end_test_case()

            # Get timestamps information
            timestamps = test_tracker.get_timestamps()

            # Check results and generate report
            result = JobManager.check_result(job_id, test_config, timestamps, full_test_name)
            assert result["success"], f"Job execution failed: {job_id}"

        except Exception as e:
            test_tracker.end_test_case()
            raise e


if __name__ == "__main__":
    """Run benchmark tests"""
    pytest.main([__file__, "-v"])
