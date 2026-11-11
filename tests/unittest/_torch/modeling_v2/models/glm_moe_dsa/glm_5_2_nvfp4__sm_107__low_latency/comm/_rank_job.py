# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one of this target's collective entries' rank jobs from pytest, and report what it did.

The matrix of an entry, ``_<entry>_op_matrix.py`` beside this file, is its own launcher: one rank
per device, its own process group, and a deadline it enforces by killing that group, which keeps a
wedged Lamport exchange from taking the calling run down with it. This only selects the devices,
starts the launcher in a fresh interpreter (it must carry no CUDA or torch.distributed state of
its own), and turns its exit code into an assertion.

The ranks import ``tensorrt_llm`` from where this pytest process finds it, so a source checkout
and an installed wheel both work.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

WORLD_SIZE = 4
"""One node's ranks. The c1 world of 16 ranks on 4 nodes is started by hand (``_mnnvl_ranks``)."""

_LAUNCHER_GRACE_S = 300
"""Headroom over the matrix's own deadline, so its message wins the race."""


def _devices() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [d for d in visible.split(",") if d.strip()]
    else:
        import torch

        devices = [str(i) for i in range(torch.cuda.device_count())]
    assert len(devices) >= WORLD_SIZE, (
        f"this matrix runs at world size {WORLD_SIZE}; only {len(devices)} device(s) are visible"
    )
    return ",".join(devices[:WORLD_SIZE])


def _load_launcher_constants(launcher: Path):
    """Read the matrix's module-level deadline without importing it by name."""
    spec = importlib.util.spec_from_file_location(launcher.stem, launcher)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(entry: str) -> None:
    """Run ``_<entry>_op_matrix``'s launcher over ``WORLD_SIZE`` devices."""
    launcher = Path(__file__).resolve().with_name(f"_{entry}_op_matrix.py")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=_devices())
    package = importlib.util.find_spec("tensorrt_llm")
    assert package is not None and package.origin is not None, "tensorrt_llm is not importable"
    package_parent = str(Path(package.origin).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (package_parent, env.get("PYTHONPATH", "")) if p)
    timeout = _load_launcher_constants(launcher).DEADLINE_S + _LAUNCHER_GRACE_S
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
