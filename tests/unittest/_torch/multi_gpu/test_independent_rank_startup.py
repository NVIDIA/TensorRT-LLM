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
"""Tests for ranks that start as independent processes under Wide-EP fault tolerance.

Each rank is a one-process MPI world that builds MnnvlDist's host collectives and an NVLinkOneSided
workspace. Run directly, this file runs one rank.
"""

import json
import os
import pickle  # nosec B403
import secrets
import subprocess  # nosec B404
import sys
import threading
import time
from unittest.mock import Mock

import pytest
import torch

WORLD = 4
JOIN_TIMEOUT_S = 30.0


def _rank_main(rank: int, world_size: int, incarnation: int) -> dict:
    from mpi4py import MPI

    from tensorrt_llm._torch.distributed import Distributed
    from tensorrt_llm._torch.distributed.host_exchange import MnnvlDist
    from tensorrt_llm._torch.distributed.membership import startup
    from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
    from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
    from tensorrt_llm._utils import mpi_comm
    from tensorrt_llm.mapping import Mapping

    errors = []

    def check(ok: bool, what: str) -> None:
        if not ok:
            errors.append(f"rank {rank}: {what}")

    check(MPI.COMM_WORLD.Get_size() == 1, "the rank is not its own MPI world")
    torch.cuda.set_device(rank)
    # Stand in for the membership client that takes the startup collectives' reports.
    startup._state.client = Mock()
    store = startup._startup_store()
    if incarnation:
        # Replay the startup that the original rank 0 recorded, as worker_node_main does.
        published = store.get(startup.AGREED_STARTUP_KEY)
        startup._state.record.replay(pickle.loads(published))  # nosec B301
    elif rank == 0:
        startup._state.record.record()
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        tp_size=world_size,
        moe_ep_size=world_size,
        enable_attention_dp=True,
        gpus_per_node=WORLD,
    )
    startup.join_startup_group(mapping)
    dist = Distributed.get(mapping)
    check(isinstance(dist, MnnvlDist), "the startup group selects the host collectives over MNNVL")
    check(dist.host_exchange is not None, "host collectives over MNNVL memory")
    hidden, top_k, capacity, slots_per_rank = 256, 2, 8, 2
    comm = NVLinkOneSided(
        mapping=mapping,
        num_slots=world_size * slots_per_rank,
        top_k=top_k,
        max_num_tokens_per_rank=capacity,
        hidden_size=hidden,
        dtype=torch.bfloat16,
        enable_wide_ep_fault_tolerance=True,
    )
    # After the startup record closes, a replacement's host collectives reach only itself.
    Distributed.startup_record.close()
    if rank == 0:
        startup.publish_startup_record(store)
    active = [rank] if incarnation else list(range(world_size))
    check(
        dist.tp_allgather(rank) == [r if r in active else None for r in range(world_size)],
        "host collective",
    )
    check(dist.allreduce(rank + 1) == sum(r + 1 for r in active), "host reduction")
    comm.bind_execution_state(
        MoeExecutionState(EPGroupHealth(world_size), comm.ep_rank, torch.device("cuda"))
    )
    check(comm.workspace.shape[0] == world_size, "workspace layout")
    if not incarnation:
        counts = [capacity - r % 2 for r in range(world_size)]
        generator = torch.Generator().manual_seed(rank)
        token = torch.arange(counts[rank])[:, None]
        # Each token goes to both slots of one rank, so each combined row is that rank's output.
        slots = ((token + rank) % world_size * slots_per_rank + torch.arange(top_k)).int().cuda()
        weights = torch.rand(counts[rank], top_k, generator=generator).cuda()
        x = torch.randn(counts[rank], hidden, generator=generator).bfloat16().cuda()
        mpi_comm().barrier()
        recv = comm.dispatch(x, None, slots, weights, counts)
        unrouted = (recv[2] < 0).all(dim=1, keepdim=True)
        output = torch.where(unrouted, float("nan"), recv[0] * 2)
        combined = comm.combine(output, all_rank_max_num_tokens=max(counts))
        torch.cuda.synchronize()
        check(torch.equal(combined, x * 2), "dispatch and combine")
        mpi_comm().barrier()
    comm.destroy()
    return {"rank": rank, "errors": errors}


def _spawn(rank: int, identity: str, port: int, secret: str) -> subprocess.Popen:
    from tensorrt_llm._torch.distributed.membership import rank_env

    env = rank_env(os.environ)
    env.update(
        TLLM_FAULT_TOLERANCE_RANK=identity,
        TLLM_FAULT_TOLERANCE_RENDEZVOUS=f"127.0.0.1:{port}",
        TLLM_FAULT_TOLERANCE_SHARED_SECRET=secret,
    )
    return subprocess.Popen(  # nosec B603
        [sys.executable, __file__, identity], env=env, stdout=subprocess.PIPE, text=True
    )


def _results(processes, timeout_s=480):
    """Return each rank's result, ending every rank on a timeout or a failed rank."""
    results = []
    try:
        for process in processes:
            out, _ = process.communicate(timeout=timeout_s)
            assert process.returncode == 0, f"a rank exited with {process.returncode}"
            results.append(json.loads(out.strip().splitlines()[-1]))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    return results


@pytest.fixture
def rendezvous(monkeypatch):
    """Host the rendezvous store and the membership authority, and return the port and secret.

    At teardown, the authority aborts the deployment as when rank 0's process ends, and exits.
    """
    from torch.distributed import TCPStore

    from tensorrt_llm._torch.distributed.membership import (
        LIVENESS_KEY,
        SHARED_SECRET_ENV,
        Authority,
        AuthorityProcess,
        SignedStore,
        open_node_supervisor_connection,
    )

    secret = secrets.token_hex(32)
    monkeypatch.setenv(SHARED_SECRET_ENV, secret)
    tcp_store = SignedStore(TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False))
    parent_read, parent_write = os.pipe()
    authority = AuthorityProcess(WORLD, WORLD, tcp_store, Authority(WORLD), parent=parent_read)
    thread = threading.Thread(target=authority.run, kwargs={"host": "127.0.0.1"}, daemon=True)
    thread.start()
    supervisor = None
    try:
        deadline = time.monotonic() + 30
        while not tcp_store.check([LIVENESS_KEY]):
            assert time.monotonic() < deadline, "the membership authority did not start"
            time.sleep(0.1)
        # The authority aborts a startup whose node has no supervisor.
        supervisor = open_node_supervisor_connection(tcp_store, 0, list(range(WORLD)), "localhost")
        yield tcp_store.port, secret
    finally:
        os.close(parent_write)
        thread.join(JOIN_TIMEOUT_S)
        os.close(parent_read)
        if supervisor is not None:
            supervisor.sock.close()
    assert not thread.is_alive(), "the membership authority did not end"


@pytest.mark.parametrize("replace", [False, True], ids=["every_rank", "replacement"])
def test_independent_ranks_build_collectives(rendezvous, replace):
    """Start every rank, rank 0 recording the startup, then a replacement of rank 2 if ``replace``.

    The replacement replays the record that rank 0 published.
    """
    # The rank processes run this file without the test's import path.
    from utils.util import skip_no_mnnvl_fabric_handles

    skip_no_mnnvl_fabric_handles(WORLD)
    port, secret = rendezvous
    starts = [(range(WORLD), 0)] + ([([2], 1)] if replace else [])
    for ranks, incarnation in starts:
        identity = f"{WORLD},{WORLD},{incarnation}"
        processes = [_spawn(rank, f"{rank},{identity}", port, secret) for rank in ranks]
        for result in _results(processes):
            assert result["errors"] == []


if __name__ == "__main__":
    rank, world_size, _, incarnation = (int(field) for field in sys.argv[1].split(","))
    print(json.dumps(_rank_main(rank, world_size, incarnation)), flush=True)
    # Skip MPI_Finalize because nothing ran over MPI.
    os._exit(0)
