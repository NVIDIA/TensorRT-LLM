"""Pytest configuration for disagg tests.

Only collects tests in this directory when --disagg parameter is provided.
Provides batch job submission capability to improve parallelism.
"""

import os

import pytest
from utils.logger import logger


def pytest_addoption(parser):
    """Add disagg-specific command line options."""
    parser.addoption(
        "--disagg",
        action="store_true",
        default=False,
        help="Enable disaggregated tests collection. Example: pytest --disagg",
    )
    parser.addoption(
        "--disagg-test-list",
        action="store",
        default=None,
        help="Path to a file containing test IDs (one per line) to run. "
        "Example: pytest --disagg --disagg-test-list=testlist/testlist_gb200.txt",
    )
    parser.addoption(
        "--disagg-batch-size",
        action="store",
        type=int,
        default=None,
        help="Number of jobs to submit per batch. Default: from env DISAGG_BATCH_SIZE or 5. "
        "Set to 0 for unlimited (submit all at once). "
        "Example: pytest --disagg --disagg-batch-size=10",
    )


def pytest_collect_directory(path, parent):
    """Only collect tests in this directory when --disagg parameter is provided.

    This hook executes earliest in the collection phase to avoid loading unnecessary test files.

    Args:
        path: Current directory path
        parent: Parent collector

    Returns:
        True: Skip collection of this directory
        None: Proceed with normal collection
    """
    disagg_enabled = parent.config.getoption("--disagg", default=False)

    if not disagg_enabled:
        # No --disagg parameter, skip collection
        return True

    # With --disagg parameter, proceed with normal collection
    return None


def pytest_collection_modifyitems(config, items):
    """Filter tests based on --disagg-test-list option.

    Args:
        config: pytest config object
        items: list of collected test items
    """
    test_list_file = config.getoption("--disagg-test-list")

    if not test_list_file:
        # No filtering needed if --disagg-test-list is not provided
        return

    # Read test IDs from file
    try:
        with open(test_list_file, "r", encoding="utf-8") as f:
            # Read non-empty lines and strip whitespace
            wanted_tests = set(
                line.strip() for line in f if line.strip() and not line.strip().startswith("#")
            )
    except FileNotFoundError:
        pytest.exit(f"Error: Test list file not found: {test_list_file}")
        return
    except Exception as e:
        pytest.exit(f"Error reading test list file {test_list_file}: {e}")
        return

    if not wanted_tests:
        pytest.exit(
            f"Error: Test list file {test_list_file} is empty or contains no valid test IDs"
        )
        return

    # Filter items based on test list
    selected = []
    deselected = []

    for item in items:
        # item.nodeid is the full test identifier like:
        # "test_disagg.py::TestDisaggBenchmark::test_benchmark[deepseek-r1-fp4:1k1k:...]"
        if item.nodeid in wanted_tests:
            selected.append(item)
        else:
            deselected.append(item)

    # Apply the filtering
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected

    # Log summary
    logger.info(f"\n{'=' * 70}")
    logger.success("Test List Filter Active")
    logger.info(f"File: {test_list_file}")
    logger.info(f"Requested: {len(wanted_tests)} test(s)")
    logger.info(f"Selected:  {len(selected)} test(s)")
    logger.info(f"Deselected: {len(deselected)} test(s)")

    if len(selected) == 0:
        logger.warning("No tests matched the test list!")
        logger.warning(f"Please check that the test IDs in {test_list_file} are correct.")

    logger.info(f"{'=' * 70}\n")


class BatchManager:
    """Batch job submission manager for disagg tests.

    Automatically splits test cases into batches and submits them on-demand
    to maximize parallelism in SLURM cluster environments.

    Key features:
    - Lazy batch submission: only submits when needed
    - Configurable batch size via CLI or environment variable
    - Maintains job_id mapping for all submitted jobs
    """

    def __init__(self, batch_size=5):
        """Initialize batch manager.

        Args:
            batch_size: Number of jobs per batch. None or 0 means unlimited (submit all at once).
                       Default is 5 if not specified.
        """
        # Normalize batch_size: None, 0, or negative means unlimited
        if batch_size is None or batch_size <= 0:
            self.batch_size = None
        else:
            self.batch_size = batch_size

        self.submitted_batches = set()  # Track which batch numbers have been submitted
        self.job_mapping = {}  # Map test_id -> SLURM job_id
        self.submit_errors = {}  # Map test_id -> error message (validation/submission failures)
        self.all_configs = []  # Ordered list of all test configs

        logger.info(f"\n{'=' * 70}")
        logger.info("Batch Manager Initialized")
        if self.batch_size:
            logger.info(f"Batch size: {self.batch_size} jobs per batch")
        else:
            logger.info("Batch size: unlimited (submit all at once)")
        logger.info(f"{'=' * 70}\n")

    def add_config(self, test_config):
        """Add a test configuration to the manager.

        Called during initialization to build the ordered list of configs.

        Args:
            test_config: TestConfig object to add
        """
        self.all_configs.append(test_config)

    def get_job_id(self, test_config):
        """Get SLURM job ID for a test config, submitting batch if needed.

        This is the main entry point. It:
        1. Determines which batch the test belongs to
        2. Submits the entire batch if not already submitted
        3. Returns the job_id for this specific test

        Args:
            test_config: TestConfig object to get job_id for

        Returns:
            str: SLURM job ID, or None if submission failed
        """
        # Find the index of this config in the ordered list
        try:
            idx = next(
                i for i, c in enumerate(self.all_configs) if c.test_id == test_config.test_id
            )
        except StopIteration:
            logger.error(f"Config not found in manager: {test_config.test_id}")
            return None

        # Calculate which batch this test belongs to
        if self.batch_size:
            batch_num = idx // self.batch_size
        else:
            batch_num = 0  # All tests in one batch

        # Submit the batch if not already submitted
        if batch_num not in self.submitted_batches:
            self._submit_batch(batch_num)

        # Return the cached job_id
        return self.job_mapping.get(test_config.test_id)

    def _submit_batch(self, batch_num):
        """Submit all jobs in a specific batch.

        Args:
            batch_num: Batch number to submit (0-indexed)
        """
        from execution.executor import JobManager
        from utils.config_validator import ConfigValidator
        from utils.job_tracker import JobTracker

        # Calculate batch range
        if self.batch_size:
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.all_configs))
        else:
            start_idx = 0
            end_idx = len(self.all_configs)

        batch_configs = self.all_configs[start_idx:end_idx]

        logger.info(f"\n{'=' * 70}")
        logger.info(f"Submitting Batch {batch_num}")
        logger.info(f"Range: [{start_idx}:{end_idx}] ({len(batch_configs)} jobs)")
        logger.info(f"{'=' * 70}\n")

        # Pre-validate all configs before submission
        logger.info("Pre-validating configurations...")
        valid_configs = []
        for config in batch_configs:
            try:
                ConfigValidator.validate_test_config(config)
                valid_configs.append(config)
            except Exception as e:
                # Validation failed - mark as None and record error
                self.job_mapping[config.test_id] = None
                self.submit_errors[config.test_id] = f"Validation failed: {str(e)}"
                logger.error(f"  [FAILED] Validation failed: {config.test_id}")
                logger.error(f"     Error: {str(e)[:100]}")

        logger.info(
            f"Validation complete: {len(valid_configs)}/{len(batch_configs)} configs valid\n"
        )

        # Submit only valid configs
        success_count = 0
        for i, config in enumerate(valid_configs, 1):
            try:
                success, job_id = JobManager.submit_test_job(config)
                if success and job_id:
                    self.job_mapping[config.test_id] = job_id
                    JobTracker.record_job(job_id)  # Record job ID for cleanup
                    success_count += 1
                    logger.success(
                        f"  [{i:3d}/{len(valid_configs)}] Job {job_id} <- {config.test_id}"
                    )
                else:
                    # Submission failed - mark as None and record error
                    self.job_mapping[config.test_id] = None
                    self.submit_errors[config.test_id] = f"Job submission failed: {job_id}"
                    logger.error(f"  [{i:3d}/{len(valid_configs)}] Failed: {config.test_id}")
            except Exception as e:
                # Submission exception - mark as None and record error
                self.job_mapping[config.test_id] = None
                self.submit_errors[config.test_id] = f"Submission exception: {str(e)}"
                logger.error(f"  [{i:3d}/{len(valid_configs)}] Error: {e}")

        # Mark batch as submitted
        self.submitted_batches.add(batch_num)

        logger.info(f"\n{'=' * 70}")
        logger.success(
            f"Batch {batch_num} Complete: {success_count}/{len(valid_configs)} submitted successfully"
        )
        if len(valid_configs) < len(batch_configs):
            logger.warning(f"Skipped {len(batch_configs) - len(valid_configs)} invalid config(s)")
        logger.info(f"{'=' * 70}\n")


@pytest.fixture(scope="session")
def batch_manager(request):
    """Provide batch manager fixture for test methods.

    This session-scoped fixture creates and initializes the BatchManager
    with all collected test configs.

    Returns:
        BatchManager: Initialized batch manager instance
    """
    # Get batch size from CLI option or environment variable
    batch_size = request.config.getoption("--disagg-batch-size")
    if batch_size is None:
        env_batch_size = os.getenv("DISAGG_BATCH_SIZE")
        if env_batch_size:
            try:
                batch_size = int(env_batch_size)
            except ValueError:
                logger.warning(f"Invalid DISAGG_BATCH_SIZE: {env_batch_size}, using default 5")
                batch_size = 5
        else:
            batch_size = 5  # Default batch size

    # Create batch manager
    manager = BatchManager(batch_size=batch_size)

    # Extract all test configs from collected items
    for item in request.session.items:
        if hasattr(item, "callspec") and "test_config" in item.callspec.params:
            manager.add_config(item.callspec.params["test_config"])

    # Log statistics
    logger.info(f"Total test configs: {len(manager.all_configs)}")
    if manager.batch_size:
        total_batches = (len(manager.all_configs) + manager.batch_size - 1) // manager.batch_size
        logger.info(f"Total batches: {total_batches}")
    logger.info("")

    return manager
