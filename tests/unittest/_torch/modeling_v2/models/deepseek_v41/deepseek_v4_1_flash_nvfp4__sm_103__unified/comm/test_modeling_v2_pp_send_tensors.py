# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the pp_send_tensors certification matrix.

The matrix is ``_pp_send_recv_op_matrix.py`` beside this file, its own launcher:
one 4-rank ``mpirun`` over the first 4 visible devices of this node (PP2 x TP2,
and the same world as PP2 x DEP2), under a deadline. It drives the one-node cells
of both ``pp_send_tensors`` and ``pp_recv_tensors``. The two-tray cells (8 ranks,
PP2 x TP4, one stage per node) cannot be started from one node: their hand-run
command is in the entry's ``note``.

The same job certifies ``pp_recv_tensors`` (``test_modeling_v2_pp_recv_tensors.py``):
neither half of a hand-over runs without the other.

The matrix keeps ``mpirun`` rather than a torch.distributed rendezvous: what it
certifies is the engine's MPI-path PP communicator (``PPCommNCCL``), whose eager
send copies the tensors onto its own stream; the ProcessGroup communicator the
engine uses with MPI disabled (``PPCommTorch``) sends the caller's tensors with no copy,
so the matrix's overwrite-after-send checks are not about it.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

assert torch.cuda.is_available(), "pp_send_tensors requires CUDA devices"

_MATRIX = Path(__file__).resolve().with_name("_pp_send_recv_op_matrix.py")
FORMS = {"one_node_pp2xtp2": []}


def _matrix_constants():
    """The matrix's module-level constants, read without importing it by name (tests/ is
    not a package) and without importing tensorrt_llm (the launcher must not)."""
    spec = importlib.util.spec_from_file_location(_MATRIX.stem, _MATRIX)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_pp_send_recv_matrix(form: str) -> None:
    """Start the matrix in `form` and assert it passed."""
    constants = _matrix_constants()
    # tests/unittest/_torch/modeling_v2/models/deepseek_v41/<target>/comm -> repo root.
    repo_root = Path(__file__).resolve().parents[8]
    assert (repo_root / "tensorrt_llm").is_dir(), (
        f"expected the repo root at {repo_root}, found no tensorrt_llm/ there; "
        f"this file moved without its parents[] index following"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)
    done = subprocess.run(
        [sys.executable, str(_MATRIX), *FORMS[form]],
        env=env,
        capture_output=True,
        text=True,
        timeout=constants.DEADLINE_S + 300,
    )
    # The matrix's own summary (placement, checks, controls, timing) for the run log.
    print(
        "\n".join(
            line
            for line in done.stdout.splitlines()
            if "rank 0" in line
            or "PP_SEND_RECV" in line
            or line.startswith(("FAILURE", "[launcher]", "[rank"))
        )
    )
    assert done.returncode == 0 and "PP_SEND_RECV_OK" in done.stdout, (
        f"{_MATRIX.name} ({form}) exited {done.returncode}\n--- stdout ---\n{done.stdout}\n"
        f"--- stderr ---\n{done.stderr}"
    )


# The job owns four devices; under xdist workers would fight over them.
@pytest.mark.no_xdist
@pytest.mark.parametrize("form", list(FORMS))
def test_pp_send_op_matrix(form: str) -> None:
    run_pp_send_recv_matrix(form)
