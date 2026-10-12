# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the allreduce op's certification matrix.

Distinct from ``tests/unittest/_torch/multi_gpu/test_allreduce.py`` and not a
duplicate of it: that file covers the fusion patterns through the
``AllReduce`` module, while this covers the op itself -- the plain summing
all-reduce, strategy by strategy, over the payloads a DeepSeek-V4.1
tensor-parallel forward hands it.

The matrix is ``_allreduce_op_matrix.py`` beside this file, its own launcher:
it spawns one worker process per device for each suite, rendezvoused through
torch.distributed's env://, under a deadline it enforces by killing every
worker. One case per single-node layout of the entry's cells (world 4; world
2; world 4 cut into the pp2 stages' groups of 2), each over the first visible
devices. The two-node layouts are run by hand (see the entry's ``note``).

Started by file path in a fresh interpreter: the launcher must carry no CUDA
or torch.distributed state of its own, and the ranks reach the entry by
absolute import, so this tree needs no package context.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.comm.allreduce import (  # noqa: E501
    allreduce as op,
)

assert torch.cuda.is_available(), "allreduce requires CUDA devices"

_MATRIX = Path(__file__).resolve().with_name("_allreduce_op_matrix.py")
_LAUNCHER_GRACE_S = 300
"""Headroom over the matrix's own deadline, so its message wins the race."""

# (layout, world size, group size) of every cell that fits on one node.
LAYOUTS = sorted(
    {
        (c.spec["layout"], c.spec["world_size"], c.spec["group_size"])
        for c in op.CELLS
        if c.spec["nodes"] == 1
    }
)


def _devices(world_size: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [d for d in visible.split(",") if d.strip()]
    else:
        devices = [str(i) for i in range(torch.cuda.device_count())]
    assert len(devices) >= world_size, (
        f"this case runs at world size {world_size}; only {len(devices)} device(s) are visible"
    )
    return ",".join(devices[:world_size])


def _matrix_constants():
    """The matrix's module-level budget, read without importing it by name
    (tests/ is not a package); its tensorrt_llm imports live in the rank body."""
    spec = importlib.util.spec_from_file_location(_MATRIX.stem, _MATRIX)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Each case spawns its own ranks over the first 2 or 4 visible devices. Under
# xdist several workers would fight for the same GPUs, so this must run alone.
@pytest.mark.no_xdist
@pytest.mark.parametrize("layout", LAYOUTS, ids=[name for name, _, _ in LAYOUTS])
def test_allreduce_op_matrix(layout) -> None:
    name, world_size, group_size = layout
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=_devices(world_size))
    # The ranks import tensorrt_llm absolutely, and a source checkout is not
    # necessarily installed.
    # tests/unittest/_torch/modeling_v2/models/deepseek_v41/<target>/comm -> repo root.
    repo_root = Path(__file__).resolve().parents[8]
    assert (repo_root / "tensorrt_llm").is_dir(), (
        f"expected the repo root at {repo_root}, found no tensorrt_llm/ there; "
        f"this file moved without its parents[] index following"
    )
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)

    completed = subprocess.run(
        [sys.executable, str(_MATRIX), "--group-size", str(group_size)],
        env=env,
        capture_output=True,
        text=True,
        timeout=_matrix_constants().DEADLINE_S + _LAUNCHER_GRACE_S,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{_MATRIX.name} ({name}) exited {completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout}\n"
            f"--- stderr ---\n{completed.stderr}"
        )
    # What the run did, for `pytest -rA` to show beside the pass.
    print(f"[{name}] world_size={world_size} group_size={group_size}")
    print(
        "\n".join(
            line
            for line in completed.stdout.splitlines()
            if line.startswith(("[rank", "[launcher]", "OK"))
        )
    )
