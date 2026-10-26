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
"""PeerBuffers: every rank's stores through ``peer_ptrs`` land in the peers' buffers."""

import pickle
import platform
import sys
import traceback

import cloudpickle
import pytest
import torch
from cuda.bindings import runtime as cudart
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._torch.distributed.peer_buffers import PeerBuffers, PeerTransport
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)

WORDS = 1024


def _words(rank: int, step: int) -> torch.Tensor:
    return torch.arange(WORDS, dtype=torch.int32, device="cuda") + (rank * 1000 + step) * WORDS + 1


@torch.inference_mode()
def _run(tp_size: int, transport: str) -> bool:
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        mapping = Mapping(
            world_size=tp_size,
            tp_size=tp_size,
            rank=rank,
            gpus_per_node=torch.cuda.device_count(),
        )
        buffers = PeerBuffers.create(mapping, tp_size * WORDS * 4, PeerTransport(transport))
        assert buffers is not None and buffers.transport is PeerTransport(transport)
        assert buffers.peer_ptrs[rank] == buffers.local_ptr
        nbytes = WORDS * 4

        def copy(dst: int, src: int) -> None:
            stream = torch.cuda.current_stream().cuda_stream
            kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
            (err,) = cudart.cudaMemcpyAsync(dst, src, nbytes, kind, stream)
            assert err == cudart.cudaError_t.cudaSuccess, err

        def push():
            for ptr in buffers.peer_ptrs:
                copy(ptr + rank * nbytes, src.data_ptr())

        def read():
            got = torch.empty(tp_size, WORDS, dtype=torch.int32, device="cuda")
            for r in range(tp_size):
                copy(got[r].data_ptr(), buffers.local_ptr + r * nbytes)
            torch.cuda.synchronize()
            return got

        # Checked after the last barrier, so a mismatch on one rank cannot stall the others.
        results = [not read().any()]
        tensorrt_llm.mpi_barrier()
        src = _words(rank, 0)
        push()
        torch.cuda.synchronize()
        tensorrt_llm.mpi_barrier()
        results.append(torch.equal(read(), torch.stack([_words(r, 0) for r in range(tp_size)])))
        tensorrt_llm.mpi_barrier()

        # The same pointers inside a captured graph.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            push()
        for step in (1, 2):
            src.copy_(_words(rank, step))
            graph.replay()
            torch.cuda.synchronize()
            tensorrt_llm.mpi_barrier()
            want = torch.stack([_words(r, step) for r in range(tp_size)])
            results.append(torch.equal(read(), want))
            tensorrt_llm.mpi_barrier()
        assert all(results), results
    except Exception:
        traceback.print_exc()
        raise
    return True


def _mnnvl_supported() -> bool:
    return platform.machine().lower() == "aarch64" and MnnvlMemory.supports_mnnvl()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
@pytest.mark.parametrize("transport", ["ipc", "mnnvl"])
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_peer_buffers_carry_every_rank_stores(mpi_pool_executor, transport):
    if transport == "mnnvl" and not _mnnvl_supported():
        pytest.skip("needs fabric memory (MNNVL)")
    tp_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(_run, *zip(*[(tp_size, transport)] * tp_size))
    for result in results:
        assert result is True
