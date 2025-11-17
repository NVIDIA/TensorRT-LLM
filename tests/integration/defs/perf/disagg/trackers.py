import os
from datetime import datetime
from typing import Dict

import pandas as pd
from common import SESSION_COLLECT_CMD_TYPE, EnvManager
from executor import run_job


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
        print(
            f"🚀 Test case started: {test_name} at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def end_test_case(self):
        """Record test case end time."""
        self.end_time = datetime.now()
        if self.test_name:
            print(
                f"✅ Test case ended: {self.test_name} at {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            duration = (self.end_time - self.start_time).total_seconds()
            print(f"⏱️  Test case duration: {duration:.2f} seconds")

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
        print(f"📅 Session started: {self.start_time}")

    def end_and_collect(self):
        """Record end time and trigger information collection."""
        self.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"📅 Session ended: {self.end_time}")

        # Prepare log file path
        output_path = EnvManager.get_output_path()
        log_file = os.path.join(output_path, "session_collect.log")

        job_name = f"{EnvManager.get_slurm_job_name()}-session-collect"
        run_result = run_job(SESSION_COLLECT_CMD_TYPE,
                             job_name,
                             log_file=log_file)

        if run_result["status"]:
            # update timestamps in CSV
            self._update_csv_timestamps()
            print("📊 Session properties collected successfully")
        else:
            print(
                f"❌ Failed to collect session properties: {run_result['msg']}")

        return run_result["status"]

    def _update_csv_timestamps(self):
        """Update timestamps in CSV using pandas."""
        output_path = EnvManager.get_output_path()
        csv_file = f"{output_path}/session_properties.csv"

        if not os.path.exists(csv_file):
            print(f"   ⚠️  CSV file not found: {csv_file}")
            return

        try:
            # Read CSV
            df = pd.read_csv(csv_file)

            # Update timestamps
            df["start_timestamp"] = self.start_time
            df["end_timestamp"] = self.end_time

            # Save back
            df.to_csv(csv_file, index=False)
            print(
                f"   ✅ Timestamps updated: {self.start_time} - {self.end_time}")

        except Exception as e:
            print(f"   ❌ Failed to update timestamps: {e}")


# Global instance
session_tracker = SessionTracker()
