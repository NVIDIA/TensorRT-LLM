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
"""TP8 fabric gather bit preservation across graph replay and tag wrap.

Run with eight MPI ranks on two NVLink-connected four-GPU GB300 nodes.
"""

import time

import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch.models.musespark_open_projection_gather import ProjectionGatherWorkspace
from tensorrt_llm.mapping import Mapping


@pytest.mark.parametrize("columns", [4096, 8192])
@pytest.mark.parametrize("initial_epoch", [0, 2**32 - 4])
@torch.inference_mode()
def test_projection_gather_replay_and_wrap(columns: int, initial_epoch: int) -> None:
    """An independent CPU oracle checks raw BF16 bits, including NaN payloads."""
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 8 or not torch.cuda.is_available():
        pytest.skip("Requires eight MPI ranks and CUDA")
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Requires GB300")
    mapping = Mapping(world_size=8, tp_size=8, rank=rank, gpus_per_node=4)
    workspace = ProjectionGatherWorkspace(mapping)
    workspace.counters.fill_(initial_epoch)

    # A large shape followed by small shapes must not leave stale tagged words
    # that become visible when a later launch grows or the tag wraps.
    for tokens in (1, 8, 128, 3, 1, 128):
        local = torch.zeros(tokens, columns // 8, dtype=torch.bfloat16, device="cuda")
        workspace.gather(local)
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = workspace.gather(local)
        for repeat in range(8):
            bits = (
                torch.arange(local.numel(), dtype=torch.int32).reshape(local.shape)
                + rank * 4099
                + repeat * 1031
            ).to(torch.int16)
            # Signed zero, positive/negative infinity, and distinct NaN bits.
            bits.view(-1)[:5] = torch.tensor([-32768, 32640, -128, 32705, -63], dtype=torch.int16)
            peers = comm.allgather(bits.numpy())
            expected = torch.cat([torch.from_numpy(peer) for peer in peers], dim=1)
            local.view(torch.int16).copy_(bits)
            torch.cuda.synchronize()
            comm.Barrier()
            if rank == repeat:
                time.sleep(0.002)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual.view(torch.int16).cpu(), expected, rtol=0, atol=0)
        # Ordered stream handoff is valid; independent concurrent calls are not.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            moved = workspace.gather(local)
        torch.cuda.current_stream().wait_stream(side)
        torch.testing.assert_close(moved.view(torch.int16).cpu(), expected, rtol=0, atol=0)
        del graph
        comm.Barrier()
