# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run `_draft_step_op_matrix.py` for one entry from pytest, and turn its exit code into an assertion.

The matrix owns the hard parts -- one rank per device, its own process group, a deadline it enforces by killing
every rank's session -- so this only selects the devices and starts it in a fresh interpreter, which must carry
no CUDA or torch.distributed state of its own (a pytest process that imported tensorrt_llm does). Started by
file path: the ranks reach the target by absolute import, and this tree is not a package.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

WORLD_SIZE = 4
"""Ranks of the single-node run: the 16-rank geometry needs four nodes and an external launcher."""

_LAUNCHER_GRACE_S = 300


def _devices() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [d for d in visible.split(",") if d.strip()]
    else:
        import torch

        devices = [str(i) for i in range(torch.cuda.device_count())]
    assert len(devices) >= WORLD_SIZE, (
        f"the draft-step exchange runs over {WORLD_SIZE} ranks; "
        f"only {len(devices)} device(s) are visible"
    )
    return ",".join(devices[:WORLD_SIZE])


def run(entry: str, expected_cells: int) -> None:
    """Run the matrix for `entry` over WORLD_SIZE devices, expecting it to drive `expected_cells` cells."""
    launcher = Path(__file__).resolve().with_name("_draft_step_op_matrix.py")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=_devices())
    # The ranks import tensorrt_llm absolutely: the package this process imports, installed or not.
    package = importlib.util.find_spec("tensorrt_llm")
    assert package is not None and package.origin is not None, "tensorrt_llm is not importable"
    package_parent = str(Path(package.origin).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (package_parent, env.get("PYTHONPATH", "")) if p)
    spec = importlib.util.spec_from_file_location(launcher.stem, launcher)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    completed = subprocess.run(
        [sys.executable, str(launcher), entry, str(expected_cells)],
        env=env,
        capture_output=True,
        text=True,
        timeout=module.DEADLINE_S + _LAUNCHER_GRACE_S,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{launcher.name} {entry} exited {completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout}\n"
            f"--- stderr ---\n{completed.stderr}"
        )
