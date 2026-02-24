"""Simple job and process tracker for GitLab CI cleanup."""

import os

from utils.common import EnvManager
from utils.logger import logger


class JobTracker:
    """Track SLURM job IDs and pytest PID for GitLab CI cleanup."""

    @staticmethod
    def get_jobs_file() -> str:
        """Get jobs.txt file path in output_path."""
        output_path = EnvManager.get_output_path()
        return os.path.join(output_path, "jobs.txt")

    @staticmethod
    def get_pid_file() -> str:
        """Get pytest.pid file path in output_path."""
        output_path = EnvManager.get_output_path()
        return os.path.join(output_path, "pytest.pid")

    @staticmethod
    def record_pid():
        """Record pytest main process PID to pytest.pid file."""
        pid = os.getpid()
        pid_file = JobTracker.get_pid_file()
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(pid_file), exist_ok=True)

            # Write PID
            with open(pid_file, "w") as f:
                f.write(f"{pid}\n")
                f.flush()

            logger.info(f"Recorded pytest PID: {pid} -> {pid_file}")
        except Exception as e:
            logger.warning(f"Failed to record PID: {e}")

    @staticmethod
    def record_job(job_id: str):
        """Append SLURM job ID to jobs.txt file.

        Args:
            job_id: SLURM job ID to record
        """
        jobs_file = JobTracker.get_jobs_file()
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(jobs_file), exist_ok=True)

            # Append job ID
            with open(jobs_file, "a") as f:
                f.write(f"{job_id}\n")
                f.flush()

            logger.debug(f"Recorded SLURM job: {job_id}")
        except Exception as e:
            logger.warning(f"Failed to record job ID {job_id}: {e}")
