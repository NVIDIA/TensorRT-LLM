# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for this target's allreduce certification matrix.

Distinct from ``tests/unittest/_torch/multi_gpu/test_allreduce.py``: that file
covers the fusion patterns through the ``AllReduce`` module, while this covers
the plain-sum op this target calls, cell by cell.

The matrix is ``_allreduce_op_matrix.py`` beside this file, which is its own
launcher. This starts it the way ``comm/_rank_job.run`` starts the catalog's
collective matrices -- same device selection, same budget arithmetic, a fresh
interpreter -- but by this file's path: ``run`` resolves a launcher beside
itself, and this one lives beside the target it certifies.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["../../../../comm"]
import _rank_job  # noqa: E402 — path above

assert torch.cuda.is_available(), "allreduce requires CUDA devices"

_LAUNCHER = Path(__file__).resolve().with_name("_allreduce_op_matrix.py")


# Spawns its own 4 rank processes over every visible device. Under xdist
# several workers would fight for the same GPUs, so this must run alone.
# Unlike the catalog's matrices this file is not under modeling_v2/comm, which
# the single-GPU lists exclude by path, so it states its own device floor.
@pytest.mark.no_xdist
@pytest.mark.skipif(
    torch.cuda.device_count() < _rank_job.WORLD_SIZE,
    reason=f"the allreduce matrix runs {_rank_job.WORLD_SIZE} ranks, one per device",
)
def test_allreduce_op_matrix() -> None:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=_rank_job._devices())
    # The ranks import tensorrt_llm absolutely and a source checkout is not
    # necessarily installed. comm -> <target> -> nemotron_h -> models ->
    # modeling_v2 -> _torch -> unittest -> tests -> repo root.
    repo_root = Path(__file__).resolve().parents[8]
    assert (repo_root / "tensorrt_llm").is_dir(), (
        f"expected the repo root at {repo_root}, found no tensorrt_llm/ there; "
        f"this file moved without its parents[] index following"
    )
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)
    constants = _rank_job._load_launcher_constants(_LAUNCHER)
    completed = subprocess.run(
        [sys.executable, str(_LAUNCHER)],
        env=env,
        capture_output=True,
        text=True,
        timeout=constants.DEADLINE_S + _rank_job._LAUNCHER_GRACE_S,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{_LAUNCHER.name} exited {completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout}\n"
            f"--- stderr ---\n{completed.stderr}"
        )
