# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the allgather op's certification matrix.

Distinct from ``tests/unittest/_torch/multi_gpu/test_allreduce.py`` and not a
duplicate of it: that file covers the *fusion patterns* through the
``AllReduce`` module, while this covers the op itself cell by cell. The names
are kept apart so review does not read one as a copy of the other.

The matrix is ``_allgather_op_matrix.py`` beside this file, its own
multi-rank launcher (see its docstring for why a collective is not driven
from inside pytest). Each case starts it in a fresh interpreter over the
first ``world_size`` visible devices, with one of the two symbols the entry
routes to, and turns its exit code into an assertion -- the shape of
``_rank_job.run``, plus the ``--transport`` argument the matrix takes.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allgather import allgather

assert torch.cuda.is_available(), "allgather requires CUDA devices"

_MATRIX = Path(__file__).resolve().with_name("_allgather_op_matrix.py")
_LAUNCHER_GRACE_S = 300
"""Headroom over the matrix's own deadline, so its message wins the race."""

WORLD_SIZES = (2, 4)
"""dep2 (Step-5's disaggregated world) and dep4, the attention-DP worlds the cells hold."""


def _matrix_constants():
    """Read the matrix's module-level constants without importing it by name.

    Executing it at module scope pulls in nothing but torch; its rank body
    imports tensorrt_llm only inside the rank processes.
    """
    spec = importlib.util.spec_from_file_location(_MATRIX.stem, _MATRIX)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_cell_is_at_a_world_this_file_runs() -> None:
    """A cell whose world no case below starts would never be driven."""
    worlds = {cell.spec.get("world", 4) for cell in allgather.CELLS}
    assert worlds <= set(WORLD_SIZES), worlds
    for cell in allgather.CELLS:
        sizes = cell.spec["sizes"]
        assert sizes is None or len(sizes) == cell.spec.get("world", 4), cell.why


# Each case starts its own rank processes over the visible devices; under
# xdist several workers would fight for the same GPUs, so this must run alone
# -- the same reason the other collective entry in this repo
# (_torch/moe/kernels/serial/test_moe_alltoall.py) carries the marker.
@pytest.mark.no_xdist
@pytest.mark.parametrize("transport", ("mpi", "pg"))
@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_allgather_op_matrix(world_size: int, transport: str) -> None:
    constants = _matrix_constants()
    assert transport in constants.TRANSPORTS
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [d for d in visible.split(",") if d.strip()]
        if visible
        else [str(i) for i in range(torch.cuda.device_count())]
    )
    assert len(devices) >= world_size, (
        f"the world-{world_size} cells need {world_size} visible devices, found {len(devices)}"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(devices[:world_size]))
    # The ranks import tensorrt_llm absolutely, and a source checkout is not
    # necessarily installed. tests/unittest/_torch/modeling_v2/comm -> repo root.
    repo_root = Path(__file__).resolve().parents[5]
    assert (repo_root / "tensorrt_llm").is_dir(), (
        f"expected the repo root at {repo_root}, found no tensorrt_llm/ there; "
        f"this file moved without its parents[] index following"
    )
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)
    completed = subprocess.run(
        [sys.executable, str(_MATRIX), f"--transport={transport}"],
        env=env,
        capture_output=True,
        text=True,
        timeout=constants.DEADLINE_S + _LAUNCHER_GRACE_S,
    )
    # The per-check timings and the cell list rank 0 prints are the receipt.
    print(completed.stdout)
    if completed.returncode != 0:
        raise AssertionError(
            f"{_MATRIX.name} --transport={transport} at world {world_size} exited "
            f"{completed.returncode}\n--- stdout ---\n{completed.stdout}\n"
            f"--- stderr ---\n{completed.stderr}"
        )
