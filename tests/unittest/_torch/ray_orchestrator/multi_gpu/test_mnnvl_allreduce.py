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
"""MNNVL AllReduce under the Ray orchestrator.

The Ray counterpart of tests/unittest/_torch/multi_gpu/test_mnnvl_allreduce.py. The kernels are
identical either way; what differs is how the multicast workspace gets built. Under MPI the
per-rank CUDA memory handles are exchanged over an mpi4py communicator, but a Ray worker is not
part of an MPI job, so the exchange has to run over the TP ProcessGroup instead. These tests cover
that path end to end: MNNVL actually enabled, workspace built from a ProcessGroup, results
matching a NCCL reference.
"""

import os
from typing import Optional

import pytest
import torch

try:
    import ray
except ModuleNotFoundError:
    from tensorrt_llm import ray_stub as ray

from tensorrt_llm._torch.distributed.communicator import TorchDist
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy
from tensorrt_llm.mapping import Mapping

# Covers both kernels and forces the workspace to grow: MNNVL switches from one-shot to two-shot
# at 1 MiB of (tokens x hidden x ranks x itemsize), and the largest case needs more than the
# initially allocated buffer.
SEQ_LEN_CASES = [1, 16, 128, 2048]
HIDDEN_SIZE = 7168
DTYPE = torch.bfloat16
# Same tolerances as the MPI test: the fused kernel accumulates in bf16, so a handful of elements
# land further from an fp32 reference than a plain allreduce would.
RTOL, ATOL = 0.05, 0.15


@ray.remote(num_gpus=1)
class MnnvlAllReduceWorker:
    """Stands up the same distributed state a TensorRT-LLM Ray worker does."""

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.master_address = os.environ["MASTER_ADDR"]

        assert len(ray.get_gpu_ids()) == 1
        self.gpu = int(ray.get_gpu_ids()[0])
        from tensorrt_llm.executor.ray_gpu_worker import RayWorkerWrapper

        torch.cuda.set_device(RayWorkerWrapper.physical_to_local_id(self.gpu))

    def _create_tcp_store(self, port: Optional[int] = None) -> torch.distributed.TCPStore:
        return torch.distributed.TCPStore(
            host_name=self.master_address,
            port=port if port is not None else 0,
            world_size=self.world_size,
            is_master=(self.rank == 0),
            wait_for_workers=False,
        )

    def setup_tcp_store(self) -> int:
        assert self.rank == 0, "Only the master worker can set up the TCP store"
        self.store = self._create_tcp_store()
        return self.store.port

    def setup_distributed_env(self, port: int) -> None:
        if self.rank != 0:
            self.store = self._create_tcp_store(port)

        torch.distributed.init_process_group(
            backend="cuda:nccl,cpu:gloo",
            store=self.store,
            world_size=self.world_size,
            rank=self.rank,
        )
        self.mapping = Mapping(
            world_size=self.world_size,
            gpus_per_node=self.world_size,
            tp_size=self.world_size,
            rank=self.rank,
        )
        TorchDist(self.mapping)

    def mnnvl_supported(self) -> bool:
        """Whether the hardware can do MNNVL at all.

        Only the hardware capability, not is_mnnvl(): the test bypasses that policy gate below, so
        skipping has to key off what the machine can actually do.
        """
        from tensorrt_llm._mnnvl_utils import MnnvlMemory

        MnnvlMemory.initialize()
        return MnnvlMemory.supports_mnnvl()

    def run(self, fusion: bool) -> bool:
        from tensorrt_llm._torch.distributed import AllReduce, AllReduceParams

        # Same bypass the MPI test uses: is_mnnvl() only opts in on aarch64 and, for AUTO, only
        # when the group spans nodes. Neither holds for a single-node CI runner, so the policy
        # gate is lifted to get at the code under test. Which allocator runs still follows the
        # machine: an IMEX-provisioned NVL domain exchanges fabric handles, anything else falls
        # back to POSIX file descriptors over an IPC socket.
        os.environ["TLLM_TEST_MNNVL"] = "1"
        torch.distributed.barrier()

        allreduce = AllReduce(mapping=self.mapping, strategy=AllReduceStrategy.MNNVL, dtype=DTYPE)
        assert allreduce.mnnvl_allreduce is not None, (
            "MNNVL AllReduce was requested but is not enabled"
        )

        # The point of the Ray path: the workspace is built from the TP ProcessGroup, never from
        # an MPI communicator.
        workspace = allreduce.mnnvl_allreduce.allreduce_mnnvl_workspaces[self.mapping]
        assert isinstance(workspace["comm"], torch.distributed.ProcessGroup), (
            f"workspace comm is {type(workspace['comm'])!r}, expected a torch ProcessGroup"
        )

        eps = 1e-5
        norm_weight = torch.randn((HIDDEN_SIZE,), dtype=DTYPE, device="cuda")
        torch.distributed.broadcast(norm_weight, src=0)

        try:
            for seq_len in SEQ_LEN_CASES:
                torch.manual_seed(42 + self.rank)
                x = torch.randn((seq_len, HIDDEN_SIZE), dtype=DTYPE, device="cuda")
                residual = torch.randn((seq_len, HIDDEN_SIZE), dtype=DTYPE, device="cuda")

                reduced = x.clone()
                torch.distributed.all_reduce(reduced)

                if fusion:
                    output, residual_out = allreduce(
                        x.clone(),
                        all_reduce_params=AllReduceParams(
                            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                            residual=residual,
                            norm_weight=norm_weight,
                            eps=eps,
                        ),
                    )

                    expected_residual = reduced + residual
                    ref = expected_residual.to(torch.float32)
                    expected_norm = (ref * torch.rsqrt(ref.pow(2).mean(-1, keepdim=True) + eps)).to(
                        DTYPE
                    ) * norm_weight

                    torch.testing.assert_close(
                        residual_out, expected_residual, rtol=RTOL, atol=ATOL
                    )
                    torch.testing.assert_close(output, expected_norm, rtol=RTOL, atol=ATOL)
                else:
                    output = allreduce(x.clone())
                    torch.testing.assert_close(output, reduced, rtol=RTOL, atol=ATOL)
        finally:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return True


def _spawn_workers(world_size: int):
    runtime_env = ray.runtime_env.RuntimeEnv()
    runtime_env["env_vars"] = os.environ.copy()
    runtime_env["env_vars"].update(
        {
            "TLLM_DISABLE_MPI": "1",
            "MASTER_ADDR": "127.0.0.1",
        }
    )

    workers = [
        MnnvlAllReduceWorker.options(runtime_env=runtime_env).remote(rank, world_size)
        for rank in range(world_size)
    ]
    ray.get([w.__ray_ready__.remote() for w in workers])

    port = ray.get(workers[0].setup_tcp_store.remote())
    ray.get([w.setup_distributed_env.remote(port) for w in workers])
    return workers


@pytest.mark.gpu2
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("fusion", [True, False], ids=["fusion", "no_fusion"])
def test_mnnvl_allreduce_over_process_group(setup_ray_cluster, fusion):
    world_size = 2
    workers = _spawn_workers(world_size)

    if not all(ray.get([w.mnnvl_supported.remote() for w in workers])):
        pytest.skip("MNNVL is not supported on this machine")

    assert all(ray.get([w.run.remote(fusion) for w in workers]))
