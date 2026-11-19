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
"""Helpers of the multi-GPU MoE All-to-All tests that run each rank in its own process.

The ranks share an MNNVL workspace through a TCPStore, so that a test can kill one of them.
"""

import contextlib
import itertools
import os
import pickle
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from _torch.moe.moe_a2a_test_utils import nvlink_one_sided_for_binding
from torch.distributed import TCPStore

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.distributed.membership_mnnvl_memory import MembershipMnnvlMemory
from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm.bindings import internal
from tensorrt_llm.mapping import Mapping

EP_SIZE, CAPACITY, HIDDEN = 4, 1024, 7168
VICTIM = EP_SIZE - 1
TIMEOUT = timedelta(seconds=300)


class StoreComm:
    """Communicator subset of MembershipMnnvlMemory over a TCPStore, so that a peer can be killed.

    Calling it gathers a value from the ranks in ``members``, with None for the others.
    """

    def __init__(self, store: TCPStore, rank: int, name: str, members: range) -> None:
        self.store, self.rank, self.name, self.members, self.calls = store, rank, name, members, 0

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return EP_SIZE

    def __call__(self, value: Any) -> list[Any]:
        self.calls += 1
        self.store.set(f"{self.name}/{self.calls}/{self.rank}", pickle.dumps(value))
        return [
            pickle.loads(self.store.get(f"{self.name}/{self.calls}/{rank}"))
            if rank in self.members
            else None
            for rank in range(EP_SIZE)
        ]

    allgather = __call__

    def barrier(self) -> None:
        self(None)


def mask_words(ranks) -> list[int]:
    """Return the rank mask of fewer than 64 ``ranks`` as the kernels' four uint64 words."""
    return [sum(1 << rank for rank in ranks), 0, 0, 0]


def controls(mask: torch.Tensor, abort: torch.Tensor) -> dict:
    return {"enable_rank_mask": True, "active_rank_mask": mask, "abort_state": abort}


def dispatch(workspace, meta, rank, experts, payloads, tokens, mask, abort):
    """Dispatch [*payloads, experts] of ``tokens`` tokens per rank. Expert e is on rank e."""
    args = (experts, [*payloads, experts], workspace, meta, tokens, rank, EP_SIZE, 1, EP_SIZE)
    return torch.ops.trtllm.moe_a2a_dispatch(*args, **controls(mask, abort))


def bound_state(
    memory: MembershipMnnvlMemory, workspace: torch.Tensor, meta: torch.Tensor, rank: int
) -> MoeExecutionState:
    """Return a survivor's MoeExecutionState, bound by a minimal NVLinkOneSided on ``workspace``."""
    health = EPGroupHealth(EP_SIZE)
    state = MoeExecutionState(health, rank, workspace.device)
    comm = nvlink_one_sided_for_binding(
        workspace=workspace,
        mnnvl_mem=memory,
        workspace_lifecycle=SimpleNamespace(metainfo=meta),
        ep_rank=rank,
    )
    comm.ep_size = EP_SIZE
    comm.bind_execution_state(state)
    return state


def victim_steps(store: TCPStore) -> int:
    return int(store.add("steps", 0))


def run_victim(store: TCPStore, workspace: torch.Tensor, meta: torch.Tensor) -> None:
    # The abort request set up front makes every dispatch push its rows without waiting. Resetting
    # this rank's row before each dispatch clears the incomplete round, so each one writes into the
    # survivors' rows again.
    mask = torch.tensor(mask_words(range(EP_SIZE)), dtype=torch.uint64, device="cuda")
    abort = torch.zeros(
        int(internal.thop.MOE_A2A_ABORT_STATE_WORDS), dtype=torch.int32
    ).pin_memory()
    abort[0] = 1
    tokens = CAPACITY // EP_SIZE
    # Its tokens go to every survivor in turn.
    experts = (torch.arange(tokens, device="cuda") % VICTIM).int()[:, None]
    raw = torch.empty((tokens, 2 * HIDDEN), dtype=torch.uint8, device="cuda")
    payload = raw.view(torch.bfloat16)
    store.wait([f"captured/{survivor}" for survivor in range(VICTIM)])
    for step in itertools.count(1):
        torch.ops.trtllm.moe_a2a_initialize(workspace, meta, VICTIM, EP_SIZE, barrier=False)
        # Alternating bytes make each write change every byte it reaches, not only the first one.
        raw.fill_(0xC3 if step % 2 else 0x3C)
        dispatch(workspace, meta, VICTIM, experts, [payload], tokens, mask, abort)
        if step % 16 == 0:
            torch.cuda.synchronize()
            store.add("steps", 1)
        if step == 64:
            store.set("running", "1")


def map_workspace(rank: int, port: int) -> tuple:
    """Return the store, the memory, the workspace and the metainfo of a shared MNNVL workspace."""
    store = TCPStore("127.0.0.1", port, is_master=False, timeout=TIMEOUT)
    torch.cuda.set_device(rank)
    mapping = Mapping(EP_SIZE, rank, EP_SIZE, tp_size=EP_SIZE, moe_ep_size=EP_SIZE)
    MembershipMnnvlMemory.initialize()
    comm = StoreComm(store, rank, "world", range(EP_SIZE))
    MembershipMnnvlMemory.comm = comm
    MembershipMnnvlMemory.comm_signature = MembershipMnnvlMemory.comm_layout_signature(mapping)
    meta = NVLinkOneSided._make_workspace_layout(
        EP_SIZE, 1, CAPACITY, HIDDEN, torch.bfloat16, None, 0, rank_mask=True
    )
    memory = MembershipMnnvlMemory(
        mapping, int(meta[int(internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)])
    )
    workspace = memory.as_torch_strided_tensor(torch.uint8)
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, rank, EP_SIZE, barrier=False)
    # No peer may write into a row before its owner has initialized it.
    comm.barrier()
    return store, memory, workspace, meta


@contextlib.contextmanager
def run_workers(script: str, mode: str) -> Iterator[tuple[TCPStore, list[subprocess.Popen]]]:
    """Run ``script`` once per rank and yield once the victim runs. Kill the workers on exit."""
    if torch.cuda.device_count() < EP_SIZE:
        pytest.skip(f"Requires {EP_SIZE} local GPUs")
    MembershipMnnvlMemory.initialize()
    if not MembershipMnnvlMemory.supports_mnnvl():
        pytest.skip("Requires an NVLink fabric supported by MNNVL")
    store = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False, timeout=TIMEOUT)
    # The victim is killed, so keep the workers off any enclosing launcher's process management.
    launcher_prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(launcher_prefixes)}
    # The workers import the shared helpers of the unit tests, as pytest makes them importable.
    unittest_root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (unittest_root, env.get("PYTHONPATH")) if p)
    workers = [
        subprocess.Popen([sys.executable, script, str(rank), str(store.port), mode], env=env)
        for rank in range(EP_SIZE)
    ]
    try:
        deadline = time.monotonic() + TIMEOUT.total_seconds()
        while not store.check(["running"]):
            assert all(worker.poll() is None for worker in workers), "worker exited during setup"
            assert time.monotonic() < deadline, "victim never started"
            time.sleep(0.01)
        yield store, workers
    finally:
        store.set("done", "1")
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.wait()
