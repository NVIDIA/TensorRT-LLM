import os
from datetime import datetime
from typing import Dict

import pandas as pd

# Import JobManager from execution
from execution.executor import JobManager

from utils.common import EnvManager
from utils.logger import logger


class TestCaseTracker:
    """Test case time tracker for individual test cases."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.test_name = None

    def start_test_case(self, test_name: str):
        """Record test case start time."""
        self.test_name = test_name
        self.start_time = datetime.now()
        logger.info(
            f"Test case started: {test_name} at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def end_test_case(self):
        """Record test case end time."""
        self.end_time = datetime.now()
        if self.test_name:
            logger.success(
                f"Test case ended: {self.test_name} at {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            duration = (self.end_time - self.start_time).total_seconds()
            logger.info(f"Test case duration: {duration:.2f} seconds")

    def get_timestamps(self) -> Dict[str, str]:
        """Get formatted timestamps for CSV."""
        if not self.start_time or not self.end_time:
            current_time = datetime.now()
            return {
                "start_timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_time__sec": 0.0,
            }

        duration = (self.end_time - self.start_time).total_seconds()
        return {
            "start_timestamp": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_timestamp": self.end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time__sec": duration,
        }

    def reset(self):
        """Reset tracker for next test case."""
        self.start_time = None
        self.end_time = None
        self.test_name = None


class SessionTracker:
    """Session time tracker."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """Record start time."""
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Session started: {self.start_time}")

    def end_and_collect(self):
        """Record end time and trigger session collection.

        Uses the new sbatch-based approach for non-blocking execution.
        Submits the job and waits for completion using JobManager.
        """
        from utils.job_tracker import JobTracker

        self.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Session ended: {self.end_time}")

        # Submit session collect job (non-blocking sbatch)
        success, job_id = JobManager.submit_session_collect_job()

        if not success:
            logger.error(f"Failed to submit session collect job: {job_id}")
            return False

        # Record session collect job ID for cleanup
        JobTracker.record_job(job_id)

        # Wait for job completion (reuses wait_for_completion method)
        logger.info(f"Waiting for session collect job {job_id} to complete...")
        JobManager.wait_for_completion(
            job_id=job_id,
            timeout=7200,  # 2 hours
            test_config=None,  # No test config for session collect
            check_early_failure=False,  # Don't check early failures
        )

        # Check if log file was created (indicates success)
        output_path = EnvManager.get_output_path()
        log_file = os.path.join(output_path, "session_collect.log")

        if os.path.exists(log_file):
            # Update timestamps in CSV
            self._update_csv_timestamps()
            logger.success("Session properties collected successfully")
            logger.info(f"Session collect log: {log_file}")
            return True
        else:
            logger.error(f"Session collect log not found: {log_file}")
            return False

    def _update_csv_timestamps(self):
        """Update timestamps in CSV using pandas."""
        output_path = EnvManager.get_output_path()
        csv_file = f"{output_path}/session_properties.csv"

        if not os.path.exists(csv_file):
            logger.warning(f"CSV file not found: {csv_file}")
            return

        try:
            # Read CSV
            df = pd.read_csv(csv_file)

            # Update timestamps
            df["start_timestamp"] = self.start_time
            df["end_timestamp"] = self.end_time

            # Save back
            df.to_csv(csv_file, index=False)
            logger.success(f"Timestamps updated: {self.start_time} - {self.end_time}")

        except Exception as e:
            logger.error(f"Failed to update timestamps: {e}")


# Global instance
session_tracker = SessionTracker()
