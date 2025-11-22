"""Simplified subprocess utilities for disagg module.

This module provides simple wrappers around subprocess for executing
SLURM commands (srun, sacct, scancel, sbatch).

No complex process tree cleanup is needed because:
1. SLURM commands (srun/sacct/scancel) are simple client tools that don't spawn complex process trees
2. Actual workloads run on remote cluster nodes managed by SLURM scheduler
3. SLURM automatically handles cleanup of remote jobs when the client disconnects
"""

import subprocess
from typing import Optional


def exec_cmd(*popenargs, timeout: Optional[float] = None, **kwargs) -> int:
    """Execute command and return exit code.

    Args:
        *popenargs: Command and arguments
        timeout: Timeout in seconds
        **kwargs: Additional subprocess arguments

    Returns:
        Exit code (0 for success, non-zero for failure)

    Raises:
        subprocess.TimeoutExpired: If timeout is reached
    """
    result = subprocess.run(*popenargs, timeout=timeout, **kwargs)
    return result.returncode


def exec_cmd_with_output(*popenargs, timeout: Optional[float] = None, **kwargs) -> str:
    """Execute command and return output as string.

    Args:
        *popenargs: Command and arguments
        timeout: Timeout in seconds
        **kwargs: Additional subprocess arguments

    Returns:
        stdout as string (decoded from bytes)

    Raises:
        subprocess.CalledProcessError: If command returns non-zero exit code
        subprocess.TimeoutExpired: If timeout is reached
    """
    result = subprocess.run(
        *popenargs,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=True,
        **kwargs
    )
    return result.stdout.decode()

