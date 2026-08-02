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
"""End-to-end raw-NCCL recovery for AllGatherReduceScatter.

This file is launched directly under three MPI ranks by test_multi_gpu.py.
Keeping it in a fresh process is important: fault-tolerance mode is captured
when TensorRT-LLM is imported, and raw NCCL communicator replacement is
process-global.
"""

import torch

from tensorrt_llm._torch.distributed.nccl_fault_tolerance import NCCL_FAULT_TOLERANCE_ENABLED
from tensorrt_llm._torch.modules.fused_moe.communication.allgather_reducescatter import (
    AllGatherReduceScatter,
)
from tensorrt_llm._torch.utils import get_device_uuid
from tensorrt_llm._utils import (
    default_gpus_per_node,
    local_mpi_rank,
    mpi_comm,
    mpi_rank,
    mpi_world_size,
)
from tensorrt_llm.mapping import Mapping

_WORLD_SIZE = 3
_FULL_GROUP = [0, 1, 2]
_SURVIVORS = [0, 2]
_RECOVERY_GENERATION = 0
# Generation 0 maps to recovery rendezvous ID 2; keep cleanup on the next
# namespace. Singleton cleanup performs no peer exchange, and the survivor and
# full-group paths also have distinct communicator keys.
_CLEANUP_RENDEZVOUS_ID = 3


def _assert_full_group_bootstrap(comm: AllGatherReduceScatter, rank: int) -> None:
    """Create the full-group raw communicator through the production class."""
    hidden_states = torch.tensor(
        [[float(rank + 1), float((rank + 1) * 10)]],
        dtype=torch.float32,
        device="cuda",
    )
    token_selected_slots = torch.tensor([[rank]], dtype=torch.int32, device="cuda")

    gathered_hidden, hidden_sf, gathered_slots, final_scales = comm.dispatch(
        hidden_states,
        None,
        token_selected_slots,
        None,
        [1, 1, 1],
    )

    expected_hidden = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        dtype=torch.float32,
        device="cuda",
    )
    expected_slots = torch.tensor([[0], [1], [2]], dtype=torch.int32, device="cuda")
    torch.testing.assert_close(gathered_hidden, expected_hidden)
    torch.testing.assert_close(gathered_slots, expected_slots)
    assert hidden_sf is None
    assert final_scales is None
    torch.cuda.synchronize()


def _assert_survivor_collectives(comm: AllGatherReduceScatter, rank: int) -> None:
    """Verify real variable-size allgather and reducescatter after recovery."""
    local_hidden = {
        0: [[1.0, 10.0]],
        2: [[3.0, 30.0], [4.0, 40.0]],
    }[rank]
    local_slots = {
        0: [[0]],
        2: [[20], [21]],
    }[rank]
    hidden_states = torch.tensor(local_hidden, dtype=torch.float32, device="cuda")
    token_selected_slots = torch.tensor(local_slots, dtype=torch.int32, device="cuda")

    gathered_hidden, hidden_sf, gathered_slots, final_scales = comm.dispatch(
        hidden_states,
        None,
        token_selected_slots,
        None,
        [1, 0, 2],
    )

    expected_hidden = torch.tensor(
        [[1.0, 10.0], [3.0, 30.0], [4.0, 40.0]],
        dtype=torch.float32,
        device="cuda",
    )
    expected_slots = torch.tensor([[0], [20], [21]], dtype=torch.int32, device="cuda")
    torch.testing.assert_close(gathered_hidden, expected_hidden)
    torch.testing.assert_close(gathered_slots, expected_slots)
    assert hidden_sf is None
    assert final_scales is None

    # Rank 0 contributes 1x and rank 2 contributes 3x. A real sum
    # reducescatter therefore returns the rank-local slice of 4x the gathered
    # tensor; a local-only or stale-full-group operation cannot satisfy this.
    combined = comm.combine(gathered_hidden * float(rank + 1))
    expected_reduced = expected_hidden * 4.0
    expected_local = expected_reduced[:1] if rank == 0 else expected_reduced[1:]
    torch.testing.assert_close(combined, expected_local)
    torch.cuda.synchronize()


def main() -> None:
    if not NCCL_FAULT_TOLERANCE_ENABLED:
        raise RuntimeError("worker must import TensorRT-LLM with TLLM_FAULT_TOLERANCE_MODE=1")

    rank = mpi_rank()
    world_size = mpi_world_size()
    if world_size != _WORLD_SIZE:
        raise RuntimeError(f"expected {_WORLD_SIZE} MPI ranks, got {world_size}")

    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("test requires visible CUDA devices")
    # These helpers preserve TensorRT-LLM's normal mapping semantics both when
    # every rank sees all allocated GPUs and when the launcher remaps each rank
    # to one CUDA-visible device.
    device = local_mpi_rank()
    torch.cuda.set_device(device)

    # Support both launchers that expose all devices to every rank and those
    # that bind each rank to a single visible device. Refuse duplicate physical
    # devices before entering NCCL, where such a setup could hang.
    device_uuids = mpi_comm().allgather(get_device_uuid(device))
    if len(set(device_uuids)) != _WORLD_SIZE:
        raise RuntimeError(
            f"test requires one distinct physical CUDA device per MPI rank; got {device_uuids}"
        )

    # Keep test-only synchronization off the communicator used internally by
    # the raw NCCL rendezvous, matching the native recovery test's isolation.
    test_sync = mpi_comm().Dup()
    mapping = Mapping(
        world_size=_WORLD_SIZE,
        rank=rank,
        tp_size=_WORLD_SIZE,
        gpus_per_node=default_gpus_per_node(),
    )
    comm = AllGatherReduceScatter(mapping)

    _assert_full_group_bootstrap(comm, rank)
    test_sync.Barrier()

    if rank in _SURVIVORS:
        # Exercise the Python class API all the way through the registered
        # torch op and native raw-communicator replacement.
        comm.abort_and_reinit(_SURVIVORS, generation=_RECOVERY_GENERATION)
    else:
        # The excluded rank does not enter the survivor rendezvous. Replace
        # its half-aborted full-group state with a singleton for safe teardown.
        torch.ops.trtllm.nccl_comm_abort_and_reinit(
            _FULL_GROUP,
            [rank],
            _CLEANUP_RENDEZVOUS_ID,
        )
    test_sync.Barrier()

    if rank in _SURVIVORS:
        _assert_survivor_collectives(comm, rank)
        # Avoid rank-skewed destruction of the final survivor communicator.
        torch.ops.trtllm.nccl_comm_abort_and_reinit(
            _SURVIVORS,
            [rank],
            _CLEANUP_RENDEZVOUS_ID,
        )

    test_sync.Barrier()
    test_sync.Free()


if __name__ == "__main__":
    main()
