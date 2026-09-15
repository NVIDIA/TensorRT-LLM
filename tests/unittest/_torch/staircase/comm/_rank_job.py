# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a collective entry's rank job from pytest, and report what it did.

The entry's own launcher already owns the hard parts -- one rank per visible
device, its own process group, and a deadline it enforces by killing that
group, which is what keeps a wedged collective from taking the calling run
down with it. A collective that breaks *hangs* rather than raising, so the
deadline is load-bearing and the ``mpi_pool_executor`` fixture has none.

So this does not reimplement any of that. It selects the devices, starts the
launcher in a fresh interpreter, and turns its exit code into an assertion.

Fresh interpreter is required, not tidiness: the launcher must not have
initialized MPI, and a pytest process that has imported ``tensorrt_llm``
already has.

Everything here is started **by file path**. The launcher must not import
``tensorrt_llm`` -- that calls ``MPI_Init``, and an MPI-initialized process
cannot start ``mpirun`` at all (measured: it exits 1 with no output from any
rank) -- and the ranks it spawns reach the catalog by absolute import, so
neither half needs a package context. This tree does not have one to give:
it is tests/, not a package.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

WORLD_SIZE = 4
"""dep4's world size -- the topology these entries are certified for."""

_LAUNCHER_GRACE_S = 300
"""Headroom over the entry's own deadline, so its message wins the race."""


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
    """Read the entry's module-level budget without importing it by name.

    There is no dotted name to import it by -- this directory is not a
    package -- and only its constants are wanted. Executing the file at module
    scope is cheap and pulls in nothing but torch; the entry keeps its
    tensorrt_llm imports inside the rank body for exactly that reason.
    """
    spec = importlib.util.spec_from_file_location(launcher.stem, launcher)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(entry: str) -> None:
    """Run ``_<entry>_op_matrix``'s launcher over ``WORLD_SIZE`` devices."""
    launcher = Path(__file__).resolve().with_name(f"_{entry}_op_matrix.py")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=_devices())

    # The ranks import tensorrt_llm absolutely, and a source checkout is not
    # necessarily installed. tests/unittest/_torch/staircase/comm -> repo root.
    repo_root = Path(__file__).resolve().parents[5]
    assert (repo_root / "tensorrt_llm").is_dir(), (
        f"expected the repo root at {repo_root}, found no tensorrt_llm/ there; "
        f"this file moved without its parents[] index following"
    )
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)

    # Read the budget off the entry rather than restating it: an entry that
    # raises its own deadline would otherwise be killed by this one first,
    # and the message a reader needs ("wedged") would be lost. reducescatter
    # runs a second, separately capped job after the main one.
    constants = _load_launcher_constants(launcher)
    timeout = constants.DEADLINE_S + getattr(constants, "WEDGE_CAP_S", 0) + _LAUNCHER_GRACE_S

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
