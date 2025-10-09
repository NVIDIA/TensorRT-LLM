# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pickle
import sys

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch.distributed import AllReduceStrategy

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)


def run_single_rank(dtype, strategy, message_size):
    import numpy as np
    import torch
    try:
        from cuda.bindings import driver as cuda
    except ImportError:
        from cuda import cuda

    import tensorrt_llm
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    torch.set_printoptions(threshold=10000)

    lowprecision_fp16_result_dict = {
        2: [0.0, 8.0, 16.0, 22.86, 32.0, 41.16, 45.72, 54.84, 64.0],
        4: [0.0, 64.0, 128.0, 182.9, 256.0, 329.2, 365.8, 438.8, 512.0],
        8: [0.0, 512.0, 1024.0, 1463.0, 2048.0, 2634.0, 2926.0, 3510.0, 4096.0]
    }

    lowprecision_bf16_result_dict = {
        2: [0.0, 8.0, 16.0, 22.875, 32.0, 41.25, 45.75, 54.75, 64.0],
        4: [0.0, 64.0, 128.0, 183, 256.0, 330, 366, 438, 512.0],
        8: [0.0, 512.0, 1024.0, 1464.0, 2048.0, 2640.0, 2928.0, 3504.0, 4096.0]
    }

    raw_result_dict = {
        2: [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0],
        4: [0.0, 64.0, 128.0, 192.0, 256.0, 320.0, 384.0, 448.0, 512.0],
        8: [0.0, 512.0, 1024.0, 1536.0, 2048.0, 2560.0, 3072.0, 3584.0, 4096.0]
    }

    def generate_reps_array(size):
        pattern = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        reps = -(-size // len(pattern))
        tiled_array = np.tile(pattern, reps)
        result_array = tiled_array[:size]
        return result_array

    class TestLowPrecisionAllreduce:

        def __init__(self,
                     message_size,
                     dtype=torch.float32,
                     strategy=AllReduceStrategy.AUTO):
            self.rank = tensorrt_llm.mpi_rank()
            self.world_size = tensorrt_llm.mpi_world_size()
            self.dtype = dtype
            self.strategy = strategy
            self.message_size = message_size

            torch.cuda.set_device(self.rank)
            self.mapping = Mapping(
                world_size=self.world_size,
                tp_size=self.world_size,
                rank=self.rank,
            )

            self.allreduce = AllReduce(mapping=self.mapping,
                                       strategy=self.strategy).cuda()

            self.input_tensors = []
            for i in range(self.world_size):
                reps_array = generate_reps_array(message_size)
                tmp_tensor = torch.from_numpy(reps_array).reshape(-1, 1024).to(
                    dtype=self.dtype, device=f'cuda:{self.rank}')
                self.input_tensors.append(tmp_tensor)

        def test(self, mode="acc"):
            if mode == "acc":
                # Testing for accuracy
                iter_num = 3
                input_tensor = self.input_tensors[self.rank]
                reference_raw_tensor = input_tensor.clone()
                for i in range(iter_num):
                    reference_raw_tensor = reference_raw_tensor * self.world_size

                for i in range(iter_num):
                    output_tensor = self.allreduce(input_tensor)
                    input_tensor = output_tensor

                reference_raw_tensor_fp32 = reference_raw_tensor.to(
                    torch.float32)
                reference_raw_tensor_np = reference_raw_tensor_fp32.cpu().numpy(
                ).flatten()
                output_fp32 = output_tensor.to(torch.float32)
                output_np = output_fp32.cpu().numpy().flatten()

                if np.array_equal(output_np, reference_raw_tensor_np):
                    print(f"Rank {self.rank}: fall back to normal allreduce")
                    return

                # Prepare the reference result
                if self.dtype == torch.float16:
                    tmp_lp_result = lowprecision_fp16_result_dict[
                        self.world_size]
                elif self.dtype == torch.bfloat16:
                    tmp_lp_result = lowprecision_bf16_result_dict[
                        self.world_size]

                tmp_raw_result = raw_result_dict[self.world_size]
                tmp_result_dict = dict(zip(tmp_raw_result, tmp_lp_result))

                keys = np.array(list(tmp_result_dict.keys()))
                values = np.array([tmp_result_dict[key] for key in keys])
                unique_keys, positions = np.unique(reference_raw_tensor_np,
                                                   return_inverse=True)
                ref_result = values[np.searchsorted(
                    keys, unique_keys)[positions]].astype(np.float16)

                # Find differences between arrays
                mask_diff = output_np != ref_result
                indices_diff = np.where(mask_diff)
                diff_values_output = output_np[indices_diff]
                ref_result[indices_diff]
                if not np.any(mask_diff):
                    print("No differences found. Test passed!")
                    assert np.all(output_np == ref_result), "have some diff"
                    print(
                        f"Rank {self.rank}: test pass world_size {self.world_size} message_size = {self.message_size} dtype = {self.dtype}"
                    )
                else:
                    total_diffs = np.sum(mask_diff)
                    print(
                        f"Rank {self.rank}: Found {total_diffs} differences. indices_diff = {indices_diff} output_np = {output_np[-1024*1024:]} ref_result = {ref_result[-1024*1024:]}"
                    )

                    # Check if total differences is multiple of 8
                    diff_num_is_wrong = (total_diffs % 8 != 0)
                    # Check if all diff_values_output are in tmp_raw_result
                    not_in_list = [
                        val for val in diff_values_output
                        if val not in tmp_raw_result
                    ]

                    if len(not_in_list) > 0 or diff_num_is_wrong:
                        raise ValueError(f"test failed")
                    else:
                        print(
                            f"Rank {self.rank}: Found diff_num_is_wrong {diff_num_is_wrong} , but is reasonable"
                        )
            elif mode == "perf":
                # Performance testing mode
                iter_num = 20
                warmup_iter_num = 10
                _, start = cuda.cuEventCreate(0)
                _, stop = cuda.cuEventCreate(0)
                stream = torch.cuda.current_stream()
                input_tensor = self.input_tensors[self.rank]
                tensorrt_llm.mpi_barrier()
                for _ in range(warmup_iter_num):
                    output_tensor = self.allreduce(input_tensor)
                torch.cuda.synchronize()
                cuda.cuEventRecord(start, stream.cuda_stream)
                for i in range(iter_num):
                    output_tensor = self.allreduce(input_tensor)
                cuda.cuEventRecord(stop, stream.cuda_stream)
                torch.cuda.synchronize()
                _, ms = cuda.cuEventElapsedTime(start, stop)

                # Calculate bandwidth
                time_in_seconds = ms / 1000.0
                avg_time_per_iter = time_in_seconds / iter_num

                # Determine datatype size in bytes
                if self.dtype == torch.float32:
                    datatype_bytes = 4
                elif self.dtype == torch.float16 or self.dtype == torch.bfloat16:
                    datatype_bytes = 2

                # Calculate algorithm bandwidth in GB/s
                total_bytes = self.message_size * datatype_bytes
                algorithm_bandwidth = (total_bytes * 2 * (self.world_size - 1) /
                                       self.world_size) / avg_time_per_iter
                algorithm_bandwidth_gb = algorithm_bandwidth / (
                    1024 * 1024 * 1024)  # Convert to GB/s

                # Print results
                print(
                    f"Rank {self.rank}: world_size={self.world_size}, message_size={self.message_size}, strategy={self.strategy}"
                )
                print(
                    f"Rank {self.rank}: Average time per iteration: {avg_time_per_iter * 1000:.3f} ms"
                )
                print(
                    f"Rank {self.rank}: Algorithm bandwidth: {algorithm_bandwidth_gb:.2f} GB/s"
                )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    print(
        f"Starting AllReduce acc test with {world_size} processes, rank={rank}, dtype={dtype}, strategy={strategy}, message_size={message_size}"
    )
    test_instance = TestLowPrecisionAllreduce(message_size=message_size,
                                              dtype=dtype,
                                              strategy=strategy)
    test_instance.test("acc")
    comm.Barrier()
    return True


# ===================== Pytest parametrized entry point =====================
@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="needs 2/4 GPUs to run this test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=["fp16", "bf16"])
@pytest.mark.parametrize("strategy", [AllReduceStrategy.LOWPRECISION],
                         ids=["lowprecision"])
@pytest.mark.parametrize(
    "message_size",
    [1024 * 1024 * x for x in [2, 4, 16, 32, 64, 132, 144]] + [64 * 70000],
    ids=lambda x: f"size{x}")
@pytest.mark.parametrize(
    "mpi_pool_executor",
    [2],  # 4, 8
    ids=["tp_size_2"],
    indirect=True)  # "tp_size_4", "tp_size_8"
def test_lowprecision_allreduce_acc(dtype, strategy, message_size,
                                    mpi_pool_executor):
    """
    Only test for accuracy. For performance testing,
    manually call TestLowPrecisionAllreduce(...).test('perf')
    """
    tp_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(dtype, strategy, message_size)] * tp_size),
    )
    for r in results:
        assert r is True
