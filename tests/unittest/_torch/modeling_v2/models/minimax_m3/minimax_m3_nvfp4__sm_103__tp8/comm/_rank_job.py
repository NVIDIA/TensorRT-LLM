# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run this target's collective rank job from pytest, and report what it did.

The same contract as ``tests/unittest/_torch/modeling_v2/comm/_rank_job.py``: the
launcher beside this file owns the per-device rank processes and the deadline
that kills a wedged collective; this only selects the devices, starts it in a
fresh interpreter (by file path: this tree is not a package) and turns its exit
code into an assertion.

World size. The tp8 target runs 8 ranks over two GB300 trays of 4 GPUs (one
NVLink domain); a single-node unit-test job cannot form that group. The
collective entries take the world size as a runtime argument and their cells'
specs do not name one (token counts, buckets, push modes, epilogue options), so
the jobs here certify every cell at each world size in ``WORLD_SIZES`` the
node's visible devices can form: 2 and 4 on a GB300 tray, 8 only on a node with
8 devices. A world size the node cannot form is skipped with that reason, never
emulated; the 8-rank, two-tray group itself is exercised only by the target's
multi-node runs.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

TARGET_WORLD_SIZE = 8
"""tp8's world size: 8 ranks over two GB300 trays, which one node's job cannot form."""

WORLD_SIZES = (2, 4, TARGET_WORLD_SIZE)
"""The world sizes the jobs certify the (world-independent) cells at, where the node has that many devices."""

_LAUNCHER_GRACE_S = 300
"""Headroom over the launcher's own deadline, so its message wins the race."""


def visible_devices() -> list:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [d for d in visible.split(",") if d.strip()]
    import torch

    return [str(i) for i in range(torch.cuda.device_count())]


def require_world(world_size: int) -> str:
    """The first ``world_size`` visible devices; skips (with the reason) when the node has fewer."""
    devices = visible_devices()
    if len(devices) < world_size:
        where = (
            "the tp8 target's own 8-rank group spans two trays and is exercised only by its multi-node runs"
            if world_size == TARGET_WORLD_SIZE
            else "run this job on a node with more devices"
        )
        pytest.skip(
            f"{world_size} ranks need {world_size} devices; {len(devices)} visible on this node ({where})"
        )
    return ",".join(devices[:world_size])


def _load_launcher_constants(launcher: Path):
    """Read the launcher's module-level budget without importing it by name (it imports only torch at
    module scope)."""
    spec = importlib.util.spec_from_file_location(launcher.stem, launcher)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(entry: str, world_size: int) -> str:
    """Run ``_<entry>_op_matrix``'s launcher over ``world_size`` devices (skipped when the node has fewer); returns
    its stdout."""
    assert world_size in WORLD_SIZES, world_size
    launcher = Path(__file__).resolve().with_name(f"_{entry}_op_matrix.py")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=require_world(world_size))

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
            f"{launcher.name} exited {completed.returncode} at world size {world_size}\n"
            f"--- stdout ---\n{completed.stdout}\n"
            f"--- stderr ---\n{completed.stderr}"
        )
    return completed.stdout
