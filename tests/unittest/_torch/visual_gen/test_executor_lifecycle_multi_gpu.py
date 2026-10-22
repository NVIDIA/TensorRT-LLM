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

import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.visual_gen import executor as executor_module
from tensorrt_llm._torch.visual_gen.executor import DiffusionRemoteClient

_COLD_SPAWN_TIMEOUT = 120.0


def _process_state(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    return stat.rsplit(")", maxsplit=1)[1].split()[0]


def _pause() -> None:
    # Keep the fixture process alive without polling or holding a Python lock.
    # SIGKILL terminates the process while blocked and never returns; the loop
    # only handles an unrelated caught signal that returns normally.
    while True:
        signal.pause()


def _gpu_bound_worker(
    rank: int,
    parent_pid: int,
    ready_queue,
) -> None:
    executor_module._start_coordinator_watchdog(parent_pid)
    torch.cuda.set_device(rank)
    # Keep a live CUDA allocation on each device while the parent injects a
    # process failure. NCCL is intentionally not initialized here: killing a
    # rank inside an active NCCL group can wedge NVIDIA UVM teardown and poison
    # the shared CI node, which tests driver recovery rather than this client's
    # worker-containment behavior.
    allocation = torch.empty(1024, device=f"cuda:{rank}")
    torch.cuda.synchronize(rank)
    ready_queue.put(rank)
    ready_queue.close()
    ready_queue.join_thread()
    assert allocation.is_cuda
    _pause()


@pytest.mark.gpu4
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="native parent monitoring and /proc are Linux-specific",
)
def test_sigkill_one_worker_contains_real_multi_gpu_group() -> None:
    world_size = 4
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")

    context = executor_module._get_mp_context("spawn")
    ready_queue = context.Queue()
    parent_pid = os.getpid()
    workers = [
        context.Process(
            target=_gpu_bound_worker,
            args=(
                rank,
                parent_pid,
                ready_queue,
            ),
        )
        for rank in range(world_size)
    ]
    for worker in workers:
        worker.start()

    try:
        ready_deadline = time.monotonic() + _COLD_SPAWN_TIMEOUT
        ready_ranks = {
            ready_queue.get(timeout=max(0.0, ready_deadline - time.monotonic()))
            for _ in range(world_size)
        }
        assert ready_ranks == set(range(world_size))

        failed_worker = workers[0]
        assert failed_worker.pid is not None
        os.kill(failed_worker.pid, signal.SIGKILL)
        deadline = time.monotonic() + 10.0
        while _process_state(failed_worker.pid) != "Z" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert _process_state(failed_worker.pid) == "Z"

        client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
        client.worker_processes = workers
        client._worker_spawner = executor_module._WorkerProcessSpawner(workers)
        client._ext_worker_thread = None
        client._monitor_worker_liveness = True
        client._worker_failure = None
        client._shutdown_started = False
        client._request_to_send = None
        client.shutdown_event = threading.Event()
        client.response_event = threading.Event()

        deadline = time.monotonic() + 10.0
        while client._worker_failure is None and time.monotonic() < deadline:
            worker_failure = client._check_worker_liveness()
            if worker_failure is not None:
                client._abort_worker_group(worker_failure)
            time.sleep(0.001)

        assert client._worker_failure == (
            f"DiffusionClient: local worker processes exited: pid={failed_worker.pid}, exitcode=-9"
        )
        assert failed_worker.exitcode == -signal.SIGKILL
        for worker in workers[1:]:
            assert worker.exitcode == -signal.SIGKILL
        for worker in workers:
            assert worker.pid is not None
            assert _process_state(worker.pid) is None
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
            worker.join(timeout=10.0)
        ready_queue.close()
        ready_queue.join_thread()
