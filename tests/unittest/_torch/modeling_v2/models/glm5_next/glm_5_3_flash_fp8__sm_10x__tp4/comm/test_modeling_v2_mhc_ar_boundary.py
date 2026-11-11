# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the fused all-reduce + mHC boundary's 4-rank matrix.

The matrix is ``_mhc_ar_boundary_op_matrix.py`` beside this file: its own launcher, started here by file path in
a fresh interpreter (it must carry no CUDA, MPI or torch.distributed state), with the devices selected and its
deadline honoured, its exit code turned into the assertion.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

assert torch.cuda.is_available(), "the fused boundary requires CUDA devices"

_WORLD = 4
_GRACE_S = 300
"""Headroom over the matrix's own deadline, so its message wins the race."""


def _matrix_deadline(launcher: Path) -> int:
    """The matrix's ``DEADLINE_S``, read off the file rather than restated (its module scope imports only
    torch; tensorrt_llm is imported inside the rank body)."""
    spec = importlib.util.spec_from_file_location(launcher.stem, launcher)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DEADLINE_S


@pytest.mark.no_xdist
def test_mhc_ar_boundary_op_matrix() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [d for d in visible.split(",") if d.strip()]
        if visible
        else [str(i) for i in range(torch.cuda.device_count())]
    )
    if len(devices) < _WORLD:
        pytest.skip(f"the op runs on {_WORLD} ranks in one node; {len(devices)} device(s) visible")
    launcher = Path(__file__).resolve().with_name("_mhc_ar_boundary_op_matrix.py")
    # tests/unittest/_torch/modeling_v2/models/glm5_next/<target>/comm -> repo root
    repo_root = Path(__file__).resolve().parents[8]
    assert (repo_root / "tensorrt_llm").is_dir(), f"expected the repo root at {repo_root}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(devices[:_WORLD]))
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)
    completed = subprocess.run(
        [sys.executable, str(launcher)],
        env=env,
        capture_output=True,
        text=True,
        timeout=_matrix_deadline(launcher) + _GRACE_S,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{launcher.name} exited {completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
        )
