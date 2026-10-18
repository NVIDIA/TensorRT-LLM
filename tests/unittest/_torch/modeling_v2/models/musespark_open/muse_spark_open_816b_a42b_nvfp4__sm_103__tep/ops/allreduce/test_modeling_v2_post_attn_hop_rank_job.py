# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point: the post_attn_hop collective test as one 4-rank job.

``test_modeling_v2_post_attn_hop.py`` beside this file is a collective test:
every rank of the group runs it, so one pytest process alone only skips it. A
cluster run starts it once per rank under MPI (that is how the two-node TP8
case runs). This entry needs no MPI launch: from one pytest process it starts 4
rank processes on the first 4 visible GPUs, which rendezvous over
``torch.distributed`` (``TLLM_DISABLE_MPI=1``, the path the MNNVL workspace
helpers already take under a non-MPI launcher), each running that same
file with pytest.

The shape is ``modeling_v2/comm/_rank_job.py``'s: fresh interpreters that carry
no CUDA or communicator state of this process, each in its own session, and a
deadline that kills every session -- a collective that breaks hangs rather than
raising, so the deadline is what turns a wedge into a failure. A rank that fails
makes the job fail at once; the others are killed with it.

Each rank writes a JUnit report, and the job passes only if every rank ran the
whole file: no failure, no error, and nothing skipped beyond the
microbenchmark, so a rank that silently skipped (no group, wrong GPU) cannot
pass for one that ran.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest
import torch

WORLD_SIZE = 4
DEADLINE_S = 900
#: At least this many tests per rank: the cells alone are 8 x 3 transports.
MIN_TESTS = 30

_COLLECTIVE = Path(__file__).resolve().with_name("test_modeling_v2_post_attn_hop.py")
#: tests/unittest, whose pytest.ini puts tests/ on the path for the import hooks.
_UNITTEST_ROOT = Path(__file__).resolve().parents[7]


def _sm103_devices() -> list:
    if not torch.cuda.is_available():
        return []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = (
        [d.strip() for d in visible.split(",") if d.strip()]
        if visible
        else [str(i) for i in range(torch.cuda.device_count())]
    )
    return [
        dev
        for index, dev in enumerate(ids)
        if index < torch.cuda.device_count()
        and torch.cuda.get_device_capability(index) in ((10, 3), (10, 7))
    ]


pytestmark = pytest.mark.skipif(
    len(_sm103_devices()) < WORLD_SIZE,
    reason=f"the post_attn_hop collective needs {WORLD_SIZE} sm_103 or sm_107 GPUs",
)


def _rank_report(path: Path) -> dict:
    suite = ElementTree.parse(path).getroot()
    if suite.tag == "testsuites":
        suite = suite[0]
    return {key: int(suite.get(key, 0)) for key in ("tests", "failures", "errors", "skipped")}


@pytest.mark.no_xdist
def test_the_post_attn_hop_collective_passes_on_4_ranks(tmp_path):
    assert (_UNITTEST_ROOT / "pytest.ini").is_file(), f"no pytest.ini at {_UNITTEST_ROOT}"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    # The ranks import the tensorrt_llm this process would import (an installed
    # wheel, or a source checkout on the path), found without importing it here.
    package = importlib.util.find_spec("tensorrt_llm")
    assert package is not None and package.origin, "tensorrt_llm is not importable"
    package_parent = str(Path(package.origin).resolve().parents[1])
    env = {
        k: v
        for k, v in os.environ.items()
        # A launching srun step's PMIx / MPI namespace must not reach the ranks:
        # they would register as participants of that step, and the first hard
        # exit would take the step (and this report) down with it.
        if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))
    }
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(_sm103_devices()[:WORLD_SIZE]),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(WORLD_SIZE),
        TLLM_DISABLE_MPI="1",
        MODELING_V2_POST_ATTN_HOP_RANK_JOB="1",
        PYTHONPATH=os.pathsep.join(p for p in (package_parent, env.get("PYTHONPATH", "")) if p),
    )
    env.pop("MODELING_V2_TIMING", None)
    logs = [tmp_path / f"rank{r}.log" for r in range(WORLD_SIZE)]
    reports = [tmp_path / f"rank{r}.xml" for r in range(WORLD_SIZE)]
    children = []
    for rank in range(WORLD_SIZE):
        with open(logs[rank], "w") as log:
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-x",
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        "--noconftest",
                        "-o",
                        "addopts=-p test_common.magic_import_hooks",
                        f"--junitxml={reports[rank]}",
                        str(_COLLECTIVE),
                    ],
                    cwd=_UNITTEST_ROOT,
                    env=dict(env, RANK=str(rank), LOCAL_RANK=str(rank)),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )

    def kill_all():
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for child in children:
            child.wait()

    deadline = time.monotonic() + DEADLINE_S
    live = set(range(WORLD_SIZE))
    failure = None
    while live and failure is None:
        if time.monotonic() > deadline:
            failure = f"did not finish in {DEADLINE_S}s (wedged)"
            break
        for rank in sorted(live):
            code = children[rank].poll()
            if code is None:
                continue
            live.discard(rank)
            if code != 0:
                failure = f"rank {rank} exited {code}"
                break
        time.sleep(0.5)
    if failure is not None:
        kill_all()
        tails = "\n".join(
            f"--- rank {r} (last 60 lines) ---\n"
            + "\n".join(logs[r].read_text(errors="replace").splitlines()[-60:])
            for r in range(WORLD_SIZE)
        )
        raise AssertionError(f"the {WORLD_SIZE}-rank post_attn_hop job failed: {failure}\n{tails}")
    for rank in range(WORLD_SIZE):
        counts = _rank_report(reports[rank])
        assert counts["failures"] == counts["errors"] == 0, f"rank {rank}: {counts}"
        assert counts["skipped"] <= 1, f"rank {rank} skipped more than the benchmark: {counts}"
        assert counts["tests"] >= MIN_TESTS, f"rank {rank} ran too few tests: {counts}"
