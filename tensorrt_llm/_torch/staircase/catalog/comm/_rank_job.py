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

For the same reason the launcher is started **by file path, not by
``-m``**. ``-m`` on a module inside this package imports every parent
package on the way to it, ``tensorrt_llm`` included, and that import calls
``MPI_Init``; the launcher would then fail to start ``mpirun`` at all
(measured: it exits 1 with no output from any rank). Run by path the file
has no package context and imports nothing but torch, which is all its
launcher half needs -- the entry keeps its relative imports inside
``_run_one_rank``, and the ranks it spawns *are* started with ``-m`` so
they get one.
"""

from __future__ import annotations

import os
import subprocess
import sys
from importlib import import_module
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


def run(entry: str) -> None:
    """Run ``<entry>_test``'s launcher over ``WORLD_SIZE`` devices."""
    module = f"{__package__}.{entry}_test"
    launcher = Path(__file__).with_name(f"{entry}_test.py")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=_devices())
    # The launcher re-execs itself per rank and needs this package importable
    # from the ranks; by path it has no package context of its own to inherit.
    repo_root = Path(__file__).resolve().parents[5]
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)

    # Read the budget off the entry rather than restating it: an entry that
    # raises its own deadline would otherwise be killed by this one first,
    # and the message a reader needs ("wedged") would be lost. reducescatter
    # runs a second, separately capped job after the main one.
    entry_module = import_module(module)
    timeout = entry_module.DEADLINE_S + getattr(entry_module, "WEDGE_CAP_S", 0) + _LAUNCHER_GRACE_S

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
