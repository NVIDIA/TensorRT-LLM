# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Find the ``mpirun`` of a Wide-EP fault-tolerance job on node 0, and end the job through it.

The job is refused unless that ``mpirun`` runs with ``--enable-recovery``, without which the first
rank exit ends the job.
"""

import functools
import os
import signal
from typing import Callable, Dict, Optional

from ..logger import logger

# Newest protocol first.
_PMIX_SERVER_URI_VARS = (
    "PMIX_SERVER_URI41",
    "PMIX_SERVER_URI4",
    "PMIX_SERVER_URI3",
    "PMIX_SERVER_URI21",
    "PMIX_SERVER_URI2",
)


def _mpirun_pid(uri: str) -> Optional[int]:
    """Return the PID of the ``mpirun`` of the PRRTE job at ``uri`` if it runs on this host."""
    # PRRTE ends the URI's namespace with the PID of mpirun (``prterun-<host>-<PID>@...``).
    pid = uri.split("@", 1)[0].rsplit("-", 1)[-1]
    try:
        with open(f"/proc/{int(pid)}/comm") as comm:
            launcher = comm.read().strip()
    except (OSError, ValueError):
        return None
    return int(pid) if launcher in ("prterun", "mpirun") else None


def _check_recovery(mpirun: int) -> None:
    """Check that the command line of ``mpirun`` on this host has ``--enable-recovery``.

    Raises:
        ValueError: The command line of ``mpirun`` lacks ``--enable-recovery``.
    """
    try:
        with open(f"/proc/{mpirun}/cmdline", "rb") as cmdline:
            argv = cmdline.read().split(b"\0")
    except OSError as error:
        logger.warning(
            f"Cannot read the command line of mpirun (PID {mpirun}) to check that it has "
            f"--enable-recovery, which Wide-EP fault tolerance requires: {error!r}"
        )
        return
    if b"--enable-recovery" not in argv:
        raise ValueError(
            "Wide-EP fault tolerance requires mpirun --enable-recovery; without it the first rank "
            "exit ends the job. Pass the flag even if recovery is also set in PRRTE's parameter "
            "files."
        )


def end_job_function(environ: Dict[str, str], nodes: int) -> Optional[Callable[[], None]]:
    """Return the function that ends the job as failed through its ``mpirun``, or None.

    The function sends SIGTERM to ``mpirun``, which then exits with a failure status. On one node,
    or outside ``mpirun``, the end of the supervisor ends the job, so it may be None.

    Raises:
        ValueError: A multi-node job started by mpirun runs mpirun on another host than node 0's,
            or without ``--enable-recovery``.
    """
    if "OMPI_COMM_WORLD_RANK" not in environ:
        return None
    uri = next((environ[name] for name in _PMIX_SERVER_URI_VARS if environ.get(name)), None)
    mpirun = None if uri is None else _mpirun_pid(uri)
    if mpirun is None:
        if nodes > 1:
            raise ValueError(
                "Node 0's supervisor ends a failed job through its mpirun, but the PMIx server URI "
                f"{uri!r} names no mpirun of this host. Run `mpirun --enable-recovery` of Open "
                "MPI 5 on node 0's host, the first host of --host"
            )
        return None
    if nodes > 1:
        _check_recovery(mpirun)
    return functools.partial(os.kill, mpirun, signal.SIGTERM)
