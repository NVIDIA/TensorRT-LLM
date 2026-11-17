# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the sm_107 tp4 target's MoE-tail all-reduces.

The checks are `_moe_tail_allreduce_ranks.py` beside this file: four ranks over four devices
of one node, started by that file's own launcher in a fresh interpreter, whose exit code this
turns into an assertion. A fresh interpreter, because the launcher owns the deadline that ends
a wedged collective and must carry no CUDA or torch.distributed state of its own.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_JOB = Path(__file__).resolve().with_name("_moe_tail_allreduce_ranks.py")
#: Headroom over the job's own deadline, so its message wins the race.
_LAUNCHER_GRACE_S = 300

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 4
    or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 tp4 target's MoE-tail all-reduces need four SM107 devices",
)


def _deadline_s() -> int:
    spec = importlib.util.spec_from_file_location(_JOB.stem, _JOB)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DEADLINE_S


# Spawns 4 rank processes over the visible devices; under xdist, workers would fight for them.
@pytest.mark.no_xdist
def test_moe_tail_allreduce_matches_the_separate_ops() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [d for d in visible.split(",") if d.strip()]
        if visible
        else [str(i) for i in range(torch.cuda.device_count())]
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(devices[:4]))
    # The ranks import tensorrt_llm absolutely; a source checkout need not be installed.
    repo_root = Path(__file__).resolve().parents[8]
    assert (repo_root / "tensorrt_llm").is_dir(), f"expected the repo root at {repo_root}"
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)
    completed = subprocess.run(
        [sys.executable, str(_JOB)],
        env=env,
        capture_output=True,
        text=True,
        timeout=_deadline_s() + _LAUNCHER_GRACE_S,
    )
    assert completed.returncode == 0, (
        f"{_JOB.name} exited {completed.returncode}\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )
