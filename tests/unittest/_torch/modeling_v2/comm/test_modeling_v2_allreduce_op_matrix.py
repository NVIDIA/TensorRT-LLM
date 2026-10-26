# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the allreduce op's certification matrix.

Distinct from ``tests/unittest/_torch/multi_gpu/test_allreduce.py``: that file
covers the ``AllReduce`` module, including the MNNVL and symmetric-memory
paths it tries first. This covers the raw op the catalog entry wraps, cell by
cell.

The matrix is ``_allreduce_op_matrix.py`` beside this file: the gpt-oss and
deepseek ``RESIDUAL_RMS_NORM`` cells plus Step-5's plain sums at hidden 4096
in bf16 and fp32. Its cells are keyed by world size, so it runs once per
size; world 8 in one node needs a B200/B300. The world-8 MNNVL entry is not
collected here: it runs as one srun task per rank (see
``_mnnvl_allreduce_op_matrix.py``).

``test_allreduce_lamport_slot_after_fp32`` runs the matrix's detector of an
engine defect instead: a 16-bit one-shot call that reuses the Lamport slot an
fp32 one-shot call wrote reads stale words (the entry's note, "Lamport slot
hazard"). The entry's ``is_valid`` refuses the fp32 call that sets it up.
The detector is a strict expected failure: it passes as long as the defect
answers exactly the way it was measured, and fails the run the moment the
one-shot kernel records its clear size in a dtype-independent unit (drop the
mark and the guard then) or the run breaks in any other way.

The matrix is its own launcher and owns the deadline that kills a wedged
collective, so this only selects the devices, starts it in a fresh interpreter
(one that carries no CUDA or torch.distributed state) and turns its exit code
into an assertion -- the shape of ``_rank_job.run``, plus the ``--checks``
argument the detector needs.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

assert torch.cuda.is_available(), "allreduce requires CUDA devices"

_MATRIX = Path(__file__).resolve().with_name("_allreduce_op_matrix.py")

# tests/unittest/_torch/modeling_v2/comm -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[5]

_LAUNCHER_GRACE_S = 300
"""Headroom over the matrix's own deadline, so its message wins the race."""


def _deadline_s() -> int:
    """The matrix's own budget, read without importing tensorrt_llm."""
    spec = importlib.util.spec_from_file_location(_MATRIX.stem, _MATRIX)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DEADLINE_S


def _devices(world_size: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [d for d in visible.split(",") if d.strip()]
    else:
        devices = [str(i) for i in range(torch.cuda.device_count())]
    return ",".join(devices[:world_size])


def _run_matrix(world_size: int, *args: str) -> None:
    assert (_REPO_ROOT / "tensorrt_llm").is_dir(), (
        f"expected the repo root at {_REPO_ROOT}; this file moved without its parents[] index"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=_devices(world_size))
    # The ranks import tensorrt_llm absolutely, and a source checkout is not
    # necessarily installed.
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO_ROOT), env.get("PYTHONPATH", "")) if p
    )
    completed = subprocess.run(
        [sys.executable, str(_MATRIX), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=_deadline_s() + _LAUNCHER_GRACE_S,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{_MATRIX.name} exited {completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout}\n"
            f"--- stderr ---\n{completed.stderr}"
        )


# Each case spawns its own rank processes over the visible devices; under
# xdist several workers would fight for the same GPUs.
@pytest.mark.no_xdist
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_allreduce_op_matrix(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        # World 8 in one node exists on B200/B300 only; a GB200/GB300 tray
        # holds four GPUs.
        pytest.skip(f"{world_size} ranks in one node need {world_size} visible GPUs")
    _run_matrix(world_size)


_LAMPORT_DEFECT = (
    "rows wrong in a bf16 one-shot call "
    "three calls after an fp32 one-shot call on the same workspace"
)
"""The detector's own assertion text; only a failure carrying it is the defect."""


@pytest.mark.no_xdist
@pytest.mark.xfail(
    raises=AssertionError,
    strict=True,
    reason=(
        "engine defect: the one-shot Lamport allreduce kernel "
        "(allreduce_fusion_kernel_oneshot_lamport in allReduceFusionKernels.cu) records and "
        "consumes its clear size in elements of the calling dtype, so a 16-bit call after an "
        "fp32 call on one workspace half-clears the fp32 Lamport slot"
    ),
)
def test_allreduce_lamport_slot_after_fp32() -> None:
    """Step-5's tp4 pattern: bf16 and fp32 sums on one workspace, world 4.

    Only the detector's own failure counts as the expected one; any other
    failure of the run is re-raised as a RuntimeError, which the mark does
    not absorb.
    """
    if torch.cuda.device_count() < 4:
        raise RuntimeError("the Lamport slot detector runs at world 4")
    try:
        _run_matrix(4, "--checks=lamport")
    except AssertionError as failure:
        if _LAMPORT_DEFECT not in str(failure):
            raise RuntimeError(f"the detector failed for another reason:\n{failure}") from None
        raise
