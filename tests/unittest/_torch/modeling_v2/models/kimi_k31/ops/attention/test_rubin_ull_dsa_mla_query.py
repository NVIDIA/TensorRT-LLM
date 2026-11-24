# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's dsa_mla_query entry: real FlatMlaQuery objects on two GPUs (the
mailbox's multicast object needs two devices), each process publishing as a Helix rank; its head
block and local rows against the FP32 query within the E4M3 band, the peer's block bitwise the
peer's rows, every other head block untouched (``_k31_ull_mla_query_ranks``, a fresh interpreter
under a deadline: a broken collective hangs rather than raising)."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_mla_query import (  # noqa: E501
    dsa_mla_query as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10
    or torch.cuda.device_count() < 2,
    reason="the MLA query kernel needs SM10x GPUs, its multicast mailbox two of them",
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)

DEADLINE_S = 900  # two processes, one query kernel compile per cell each


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    path = tmp_path_factory.mktemp("mla_query") / "results.json"
    path.write_text(json.dumps({"cells": [dict(why=c.why, **c.spec) for c in CELLS]}))
    job = Path(__file__).resolve().with_name("_k31_ull_mla_query_ranks.py")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else [str(i) for i in range(torch.cuda.device_count())]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=",".join(devices[:2]),
        PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
    )
    proc = subprocess.Popen([sys.executable, str(job), str(path)], env=env,
                            start_new_session=True)  # fmt: skip
    try:
        rc = proc.wait(timeout=DEADLINE_S)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        pytest.fail(f"the two-process MLA query job did not finish in {DEADLINE_S} s")
    assert rc == 0, f"the two-process MLA query job exited with {rc}"
    out = json.loads(path.read_text())
    assert "cells" not in out, "the job wrote no results"
    return out


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, results) -> None:
    assert results[cell.why] == "ok", results[cell.why]


def test_is_valid_refuses_what_the_kernel_would_read_wrongly(results) -> None:
    assert results["is_valid"] == ["rows", "parity"]
