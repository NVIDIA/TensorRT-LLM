# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the startup of a Wide-EP fault-tolerant deployment with each rank in its own process.

Each process runs the real worker start (``start_rank`` on rank 0, ``worker_node_main`` on the
others) and the real membership authority. A fake ``worker_main`` sleeps to stand in for the model
load, calls ``wait_for_every_rank``, and serves until it is ended.
"""

import inspect
import os
import queue
import signal
import subprocess  # nosec B404
import sys
import threading
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from tensorrt_llm._torch.distributed.membership import (
    RENDEZVOUS_ENV,
    SHARED_SECRET_ENV,
    open_node_supervisor_connection,
    open_store,
    startup,
)
from tensorrt_llm._wide_ep_fault_tolerance import FAULT_TOLERANCE_RANK_ENV
from tensorrt_llm.executor import worker

pytestmark = pytest.mark.cpu_only

STARTUP_S = 600  # each process imports tensorrt_llm
RENDEZVOUS_TIMEOUT_S = 20.0  # the deployment's, scaled down for the test
# A rank failure during startup ends every process within this bound, half the membership's dead
# timeout and far below the model load time (LOAD_S).
PROMPT_EXIT_S = 30.0
LOAD_S = 3600.0
JOIN_TIMEOUT_S = 10.0
_SECRET = "ab" * 32

_WORKER = r"""
import os, sys, time

from tensorrt_llm._torch.distributed.membership import startup
from tensorrt_llm._wide_ep_fault_tolerance import wide_ep_fault_tolerance_rank

load_s, startup.RENDEZVOUS_TIMEOUT_S = float(sys.argv[1]), float(sys.argv[2])


def worker_main(**args):
    startup.start_rank(args)
    rank = wide_ep_fault_tolerance_rank().rank
    print(f"STARTED {rank} {args['marker']}", flush=True)
    if rank == 0:
        print(f"AUTHORITY {startup._state.authority.pid}", flush=True)
    time.sleep(load_s)  # the model load
    startup.wait_for_every_rank()
    print(f"SERVING {rank}", flush=True)
    time.sleep(3600)


startup.worker_main = worker_main
if os.environ["TLLM_FAULT_TOLERANCE_RANK"].startswith("0,"):
    worker_main(
        hf_model_dir=None,
        marker="rank-0-arguments",
        worker_queues=None,
        rpc_addr=None,
        hmac_key=b"",
    )
else:
    print("ENTERED", flush=True)
    startup.worker_node_main()
"""

_LAUNCHER_PREFIXES = ("OMPI_", "PMIX_", "PMI_", "PRTE_", "SLURM_")


def _rendezvous_address():
    """Return the rendezvous address and the socket that reserves its port, as the server does."""
    with mock.patch.dict(os.environ):
        for name in (SHARED_SECRET_ENV, RENDEZVOUS_ENV, FAULT_TOLERANCE_RANK_ENV):
            os.environ.pop(name, None)
        os.environ.pop("TLLM_SPAWN_PROXY_PROCESS", None)
        env, reserved = startup.ensure_rendezvous(None)
        return env[RENDEZVOUS_ENV], reserved


class Deployment:
    """Run the processes of a deployment of ``world`` ranks and queue their output lines."""

    def __init__(self, world, rendezvous_timeout_s=RENDEZVOUS_TIMEOUT_S):
        self.world = world
        self.rendezvous_timeout_s = rendezvous_timeout_s
        self.address, self._reserved = _rendezvous_address()
        self.processes = {}
        self.lines = queue.Queue()
        self.seen = []
        self._readers = []
        self._supervisor = None

    def start(self, rank, load_s=LOAD_S):
        env = {k: v for k, v in os.environ.items() if not k.startswith(_LAUNCHER_PREFIXES)}
        env.update(
            **{
                SHARED_SECRET_ENV: _SECRET,
                RENDEZVOUS_ENV: self.address,
                FAULT_TOLERANCE_RANK_ENV: f"{rank},{self.world},{self.world},0",
            },
            TRT_LLM_NO_LIB_INIT="1",
            MPI4PY_RC_INITIALIZE="false",
            CUDA_VISIBLE_DEVICES="",
        )
        process = subprocess.Popen(  # nosec B603
            [sys.executable, "-c", _WORKER, str(load_s), str(self.rendezvous_timeout_s)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.processes[rank] = process
        reader = threading.Thread(target=self._read, args=(rank, process), daemon=True)
        reader.start()
        self._readers.append(reader)
        if rank == 0:
            self._announce_node_supervisor()
        return process

    def _announce_node_supervisor(self):
        """Connect as the node supervisor of every rank, without which the authority aborts."""
        with mock.patch.dict(os.environ, {SHARED_SECRET_ENV: _SECRET}):
            self._supervisor = open_node_supervisor_connection(
                open_store(self.address, STARTUP_S), 0, list(range(self.world)), "localhost"
            )

    def _read(self, rank, process):
        for line in process.stdout:
            self.lines.put((rank, line.strip()))

    def wait_for(self, predicate, timeout_s):
        """Return the first printed (rank, line) pair for which ``predicate`` holds."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                item = self.lines.get(timeout=1)
            except queue.Empty:
                continue
            self.seen.append(item)
            if predicate(*item):
                return item
        raise AssertionError(f"timed out; printed {self.seen}")

    def wait_for_all(self, word, ranks, timeout_s=STARTUP_S):
        pending = set(ranks)
        deadline = time.monotonic() + timeout_s
        while pending:
            rank, _ = self.wait_for(
                lambda r, line: line.startswith(word) and r in pending, deadline - time.monotonic()
            )
            pending.discard(rank)

    def authority_pid(self):
        return int(next(line for _, line in self.seen if line.startswith("AUTHORITY")).split()[1])

    def statuses_within(self, ranks, timeout_s):
        """Return the exit status of each of ``ranks``, which must all end within ``timeout_s``."""
        deadline = time.monotonic() + timeout_s
        statuses = {}
        for rank in ranks:
            try:
                statuses[rank] = self.processes[rank].wait(max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                raise AssertionError(f"rank {rank} did not end within {timeout_s} s") from None
        return statuses

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.end()

    def end(self):
        for process in self.processes.values():
            if process.poll() is None:
                process.kill()
            process.wait()
        for reader in self._readers:  # each ends at its process's end of output
            reader.join(JOIN_TIMEOUT_S)
        if self._supervisor is not None:
            self._supervisor.sock.close()


def _process_ended(pid, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with open(f"/proc/{pid}/stat") as stat:
                if stat.read().rsplit(")", 1)[1].split()[0] == "Z":
                    return True  # ended, not reaped yet
        # The process may be reaped between the open and the read.
        except (FileNotFoundError, ProcessLookupError):
            return True
        time.sleep(0.2)
    return False


@pytest.mark.timeout(900)
def test_ranks_wait_for_late_rank_0():
    """Run worker_main on every rank with rank 0's arguments, even when rank 0 starts last.

    Every rank then ends with an error when the authority dies, so that the launcher ends the job.
    """
    with Deployment(3) as ranks:
        ranks.start(1, load_s=1)
        ranks.start(2, load_s=1)
        ranks.wait_for_all("ENTERED", [1, 2])
        ranks.start(0, load_s=1)
        ranks.wait_for_all("STARTED", range(3))
        assert {line for _, line in ranks.seen if line.startswith("STARTED")} == {
            f"STARTED {rank} rank-0-arguments" for rank in range(3)
        }
        ranks.wait_for_all("SERVING", range(3))
        os.kill(ranks.authority_pid(), signal.SIGKILL)
        assert set(ranks.statuses_within(range(3), PROMPT_EXIT_S).values()) == {1}


@pytest.mark.timeout(900)
@pytest.mark.parametrize("failed_rank", [0, 1])
def test_startup_failure_ends_every_process(failed_rank):
    """Abort as soon as the failed rank's connection closes, without waiting for a timeout."""
    with Deployment(3) as ranks:
        for rank in range(3):
            ranks.start(rank)
        ranks.wait_for_all("STARTED", range(3))
        authority = ranks.authority_pid()
        ranks.processes[failed_rank].kill()
        others = [rank for rank in range(3) if rank != failed_rank]
        assert set(ranks.statuses_within(others, PROMPT_EXIT_S).values()) == {1}
        assert _process_ended(authority, PROMPT_EXIT_S)


@pytest.mark.timeout(900)
def test_rank_without_rank_0_times_out():
    """End a rank with an error once rank 0's rendezvous store did not answer within the timeout."""
    with Deployment(2) as ranks:
        ranks.start(1)
        ranks.wait_for_all("ENTERED", [1])
        entered_s = time.monotonic()
        assert ranks.statuses_within([1], STARTUP_S + RENDEZVOUS_TIMEOUT_S)[1] == 1
        waited_s = time.monotonic() - entered_s
        ranks.end()
        printed = [line for _, line in ranks.seen + list(ranks.lines.queue)]
    assert waited_s >= RENDEZVOUS_TIMEOUT_S - 1.0, waited_s
    assert any("timed out" in line.lower() for line in printed), printed


def test_rank_0_forwards_every_worker_argument(monkeypatch):
    """Check that rank 0 hands every worker_main argument to the other ranks."""

    class Started(Exception):
        pass

    forwarded = {}

    def start_rank(worker_main_args):
        forwarded.update(worker_main_args)
        raise Started

    monkeypatch.setattr(startup, "start_rank", start_rank)
    monkeypatch.setattr(worker, "check_node_supervisor_launch", lambda fault_tolerance: None)
    arguments = {name: object() for name in inspect.signature(worker.worker_main).parameters}
    arguments["llm_args"] = SimpleNamespace(enable_wide_ep_fault_tolerance=True)
    with pytest.raises(Started):
        worker.worker_main(**arguments)
    assert forwarded == arguments
