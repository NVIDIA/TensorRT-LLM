# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the Hy4-Flash target's `comm/mnnvl_allreduce__r22`
certification matrix.

The cells are world 8 over two GB300 trays, which one node's four devices
cannot reach. The spool runs this file once per node of a ``--nodes 2`` job;
each node's copy spawns that node's four ranks, and the two rendezvous on
``MV2FLOW_JOB_MASTER``. The matrix is ``_mnnvl_allreduce__r22_op_matrix.py``
beside this file. Outside such a job the test skips: there is no group to
certify on.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

assert torch.cuda.is_available(), "mnnvl_allreduce requires CUDA devices"

_BODY = Path(__file__).resolve().parent / "_mnnvl_allreduce__r22_op_matrix.py"

#: Past the body's own deadline, so a wedged collective reports there first.
_DEADLINE_S = 1500


# Both nodes run this file in the same job and each spawns its own ranks over
# its own devices; under xdist several workers would fight for them.
@pytest.mark.no_xdist
def test_mnnvl_allreduce__r22_two_tray_op_matrix() -> None:
    nodes = int(os.environ.get("MV2FLOW_JOB_NODES", "1"))
    if nodes < 2:
        pytest.skip(
            "the cells are world 8 over two GB300 trays: run this file as a --nodes 2 spool job"
        )
    completed = subprocess.run([sys.executable, str(_BODY)], timeout=_DEADLINE_S)
    assert completed.returncode == 0, (
        f"the two-node rank body exited {completed.returncode}: see this node's log"
    )
