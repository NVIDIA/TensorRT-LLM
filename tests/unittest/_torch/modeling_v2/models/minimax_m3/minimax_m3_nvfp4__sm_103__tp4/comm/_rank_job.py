# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run this target's collective rank job from pytest, and report what it did.

The same contract as ``tests/unittest/_torch/modeling_v2/comm/_rank_job.py``: the
launcher beside this file owns the per-device rank processes and the deadline
that kills a wedged collective; this only selects the devices, starts it in a
fresh interpreter (by file path: this tree is not a package) and turns its exit
code into an assertion.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

WORLD_SIZE = 4
"""tp4's world size -- the topology these entries are certified for."""

_LAUNCHER_GRACE_S = 300
"""Headroom over the launcher's own deadline, so its message wins the race."""


def _devices() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [d for d in visible.split(",") if d.strip()]
    else:
        import torch

        devices = [str(i) for i in range(torch.cuda.device_count())]
    assert len(devices) >= WORLD_SIZE, (
        f"this entry is certified at world size {WORLD_SIZE}; only "
        f"{len(devices)} device(s) are visible"
    )
    return ",".join(devices[:WORLD_SIZE])


def _load_launcher_constants(launcher: Path):
    """Read the launcher's module-level budget without importing it by name (it imports only torch at
    module scope)."""
    spec = importlib.util.spec_from_file_location(launcher.stem, launcher)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(entry: str) -> str:
    """Run ``_<entry>_op_matrix``'s launcher over ``WORLD_SIZE`` devices; returns its stdout."""
    launcher = Path(__file__).resolve().with_name(f"_{entry}_op_matrix.py")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=_devices())

    # The ranks import tensorrt_llm absolutely, and a source checkout is not necessarily installed.
    # tests/unittest/_torch/modeling_v2/models/minimax_m3/<target>/comm -> repo root.
    repo_root = Path(__file__).resolve().parents[8]
    assert (repo_root / "tensorrt_llm").is_dir(), (
        f"expected the repo root at {repo_root}, found no tensorrt_llm/ there; "
        f"this file moved without its parents[] index following"
    )
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)

    constants = _load_launcher_constants(launcher)
    timeout = constants.DEADLINE_S + _LAUNCHER_GRACE_S

    completed = subprocess.run(
        [sys.executable, str(launcher)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{launcher.name} exited {completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout}\n"
            f"--- stderr ---\n{completed.stderr}"
        )
    return completed.stdout
