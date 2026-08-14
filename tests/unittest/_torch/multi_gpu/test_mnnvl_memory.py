# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import socket
import unittest

import pytest
import torch

import tensorrt_llm as tllm
from tensorrt_llm import Mapping


class TestMnnvlMemory(unittest.TestCase):

    def setUp(self):
        tllm.logger.set_level('error')
        self.world_size = tllm.mpi_world_size()
        self.rank = tllm.mpi_rank()
        # get num of task per node
        hostname = socket.gethostname()
        self.comm = tllm.mpi_comm()
        all_hostnames = self.comm.allgather(hostname)
        local_ntasks_per_node = all_hostnames.count(hostname)
        all_ntasks_per_node = self.comm.allgather(local_ntasks_per_node)
        uniform_ntasks = all(x == all_ntasks_per_node[0]
                             for x in all_ntasks_per_node)
        assert uniform_ntasks, "Not all nodes has same ntasks_per_node"
        self.local_world_size = local_ntasks_per_node
        self.local_rank = self.rank % self.local_world_size
        local_dev_count = torch.cuda.device_count()
        assert self.local_world_size <= local_dev_count, "ntasks_per_node should be less than local device count"
        torch.cuda.set_device(self.local_rank)
        tllm.MnnvlMemory.initialize()
        self.mapping = Mapping(self.world_size,
                               self.rank,
                               self.local_world_size,
                               tp_size=self.world_size)

    @staticmethod
    def align_memory(size: int):
        align_size = 2 * 1024 * 1024
        return (size + align_size - 1) // align_size * align_size

    @pytest.mark.skipif(not tllm.MnnvlMemory.supports_mnnvl(),
                        reason="Mnnvl memory is not supported on this platform"
                        )  # Skip tests on unsupported platform
    def test_mnnvl_memory(self):
        # allocate un-aligned memory
        allocate0_size = 4 * 1024 * 1024 - 3 * 1024
        mnnvl_memory0 = tllm.MnnvlMemory(self.mapping, allocate0_size)
        allocate0_size_aligned = TestMnnvlMemory.align_memory(allocate0_size)
        assert tllm.MnnvlMemory.current_mem_offset == allocate0_size_aligned

        tensor0 = mnnvl_memory0.as_torch_strided_tensor(torch.int32)
        numel_per_rank = allocate0_size // 4
        tensor0[(self.rank + 1) % self.world_size] = torch.arange(
            start=self.rank, end=self.rank + numel_per_rank, device='cuda')
        tllm.mpi_barrier()
        for r in range(self.world_size):
            torch.equal(
                tensor0[(r + 1) % self.world_size],
                torch.arange(start=r, end=r + numel_per_rank, device='cuda'))

        allocate1_size = 30 * 1024 * 1024 - 2 * 1024
        mnnvl_memory1 = tllm.MnnvlMemory(self.mapping, allocate1_size)
        allocate1_size_aligned = TestMnnvlMemory.align_memory(allocate1_size)
        assert tllm.MnnvlMemory.current_mem_offset == allocate0_size_aligned + allocate1_size_aligned
        tensor1 = mnnvl_memory1.as_torch_strided_tensor(torch.float32)
        numel_per_rank = allocate1_size // 4
        tensor1[(self.rank + 5) % self.world_size] = torch.arange(
            start=self.rank,
            end=self.rank + numel_per_rank,
            dtype=torch.float32,
            device='cuda')
        tllm.mpi_barrier()
        for r in range(self.world_size):
            torch.equal(
                tensor1[(r + 5) % self.world_size],
                torch.arange(start=r,
                             end=r + numel_per_rank,
                             dtype=torch.float32,
                             device='cuda'))
        tllm.mpi_barrier()
        del tensor0, mnnvl_memory0
        tllm.mpi_barrier()

        large_allocation2_size = 768 * 1024 * 1024
        large_mnnvl_memory2 = tllm.MnnvlMemory(self.mapping,
                                               large_allocation2_size)
        allocate2_size_aligned = TestMnnvlMemory.align_memory(
            large_allocation2_size)
        assert tllm.MnnvlMemory.current_mem_offset == allocate2_size_aligned
        assert large_mnnvl_memory2.rank_stride == (1 << 30)

        del tensor1


if __name__ == "__main__":
    unittest.main()
