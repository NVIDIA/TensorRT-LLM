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

Fresh interpreter is required, not tidiness: the launcher owns the deadline
that kills a wedged collective's process sessions, and it must carry no CUDA
or torch.distributed state of its own -- a pytest process that has imported
``tensorrt_llm`` does.

Everything here is started **by file path**: the ranks reach the catalog by
absolute import, so neither half needs a package context. This tree does not
have one to give: it is tests/, not a package.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

WORLD_SIZE = 4
"""dep4's world size -- the default topology these entries are certified for (one node)."""

_LAUNCHER_GRACE_S = 300
"""Headroom over the entry's own deadline, so its message wins the race."""


def adopt_srun_env() -> int:
    """Take this rank's identity from the srun step that started it, and
    return the node-local device it owns.

    The spawning launcher reaches only the devices of one node. A world that
    spans GB300 trays is started instead as one srun task per rank,
    ``python <matrix> --srun``, with ``MASTER_ADDR`` and ``MASTER_PORT``
    naming the first node. A failing rank exits nonzero and
    ``srun --kill-on-bad-exit=1`` stops its siblings, which is the job the
    spawning launcher's deadline does on one node.

    The device is returned rather than left in ``LOCAL_RANK`` for the rank
    body to read: on the spawning path every rank inherits one shared
    ``LOCAL_RANK`` from the launching shell, and reading it there put all
    ranks on one GPU.
    """
    for key in ("MASTER_ADDR", "MASTER_PORT"):
        assert os.environ.get(key), f"--srun needs {key} set to the first node's rendezvous"
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    return int(os.environ["SLURM_LOCALID"])


def _devices(world_size: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [d for d in visible.split(",") if d.strip()]
    else:
        import torch

        devices = [str(i) for i in range(torch.cuda.device_count())]
    assert len(devices) >= world_size, (
        f"this run is at world size {world_size}; only {len(devices)} device(s) are visible"
    )
    return ",".join(devices[:world_size])


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


def run(entry: str, world_size: int = WORLD_SIZE, directory: Path | None = None) -> None:
    """Run ``_<entry>_op_matrix``'s launcher over ``world_size`` devices of this node (its local ``mpirun``).

    A launcher that takes ``--world-size`` (the stateful entries' ``_lockstep`` launchers) also runs as one of N
    ranks started by an external launcher, ``--launcher srun --world-size N``: that is how a multi-node receipt is
    recorded, outside pytest; the others ignore the argument and size themselves from CUDA_VISIBLE_DEVICES.

    The launcher is looked up in ``directory``, by default this file's own: a target-owned collective keeps its
    launcher beside its own tests.
    """
    base = Path(__file__).resolve().parent if directory is None else Path(directory).resolve()
    launcher = base / f"_{entry}_op_matrix.py"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=_devices(world_size))

    # The ranks import tensorrt_llm absolutely, and a source checkout is not
    # necessarily installed. Hand them the checkout only when it is the
    # tensorrt_llm this process resolves (an in-tree build, or nothing
    # installed). Otherwise the installed package is already on their path,
    # and the checkout must not come first: CI runs from an unbuilt checkout,
    # whose tensorrt_llm/ has no compiled bindings, beside an installed wheel.
    # tests/unittest/_torch/modeling_v2/comm -> repo root.
    repo_root = Path(__file__).resolve().parents[5]
    installed = importlib.util.find_spec("tensorrt_llm")
    if installed is None or Path(installed.origin).resolve().parents[1] == repo_root:
        assert (repo_root / "tensorrt_llm").is_dir(), (
            f"expected the repo root at {repo_root}, found no tensorrt_llm/ there; "
            f"this file moved without its parents[] index following"
        )
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p
        )

    # Read the budget off the entry rather than restating it: an entry that
    # raises its own deadline would otherwise be killed by this one first,
    # and the message a reader needs ("wedged") would be lost. reducescatter
    # runs a second, separately capped job after the main one.
    constants = _load_launcher_constants(launcher)
    timeout = constants.DEADLINE_S + getattr(constants, "WEDGE_CAP_S", 0) + _LAUNCHER_GRACE_S

    completed = subprocess.run(
        [sys.executable, str(launcher), "--world-size", str(world_size)],
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
