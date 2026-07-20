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

"""Subprocess test for the real four-rank asynchronous consensus transport."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_PROCESS_TIMEOUT_S = 90


@pytest.mark.timeout(_PROCESS_TIMEOUT_S)
def test_async_consensus_real_mpi_four_rank_protocol() -> None:
    pytest.importorskip("mpi4py")
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not found on PATH")

    worker = Path(__file__).with_name("async_consensus_mpi_worker.py")
    repository_root = worker.parents[3]
    env = os.environ.copy()
    env.update(
        {
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(repository_root), env.get("PYTHONPATH")))
            ),
            "PYTHONUNBUFFERED": "1",
        }
    )
    completed = subprocess.run(
        [
            mpirun,
            "--allow-run-as-root",
            "--oversubscribe",
            "-np",
            "4",
            sys.executable,
            str(worker),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=_PROCESS_TIMEOUT_S - 5,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "ASYNC_CONSENSUS_MPI_OK" in output, output
