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

from argparse import ArgumentParser

# isort: off
import torch
# isort: on
from cuda import cuda, cudart

import tensorrt_llm as tllm
from tensorrt_llm import Mapping, Tensor
from tensorrt_llm._utils import OMPI_COMM_TYPE_HOST, mpi_comm
from tensorrt_llm.functional import (AllReduceParams, AllReduceStrategy,
                                     allreduce)
from tensorrt_llm.plugin.plugin import (current_all_reduce_helper,
                                        init_all_reduce_helper)
from tensorrt_llm.runtime import Session


def allreduce_benchmark(dtype: str,
                        test_range: str = "10,10000000,10",
                        no_header: bool = False):
    tllm.logger.set_level('error')
    world_size = tllm.mpi_world_size()
    rank = tllm.mpi_rank()
    local_comm = mpi_comm().Split_type(split_type=OMPI_COMM_TYPE_HOST)
    local_rank = local_comm.Get_rank()
    gpus_per_node = local_comm.Get_size()

    torch.cuda.set_device(local_rank)
    cudart.cudaSetDevice(local_rank)

    mapping = Mapping(world_size, rank, gpus_per_node, tp_size=world_size)

    if world_size == 1:
        raise RuntimeError("Benchmark must run with mpi_world_size > 1")

    torch_dtype = tllm._utils.str_dtype_to_torch(dtype)
    min_size, max_size, ratio = [int(i) for i in test_range.split(",")]
    inner_loop = 1000

    size = min_size
    dtype_size = torch.finfo(torch_dtype).bits // 8
    if mapping.rank == 0 and not no_header:
        print(
            f"{'world_size':<15}, {'dtype':<10}, {'message size':<15}, {'strategy':<15}, {'duration (ms)':<10}"
        )
    while size < max_size:
        input = torch.ones(size, dtype=torch_dtype, device="cuda")

        for strategy in [
                AllReduceStrategy.AUTO,
                AllReduceStrategy.NCCL,
                AllReduceStrategy.ONESHOT,
                AllReduceStrategy.TWOSHOT,
        ]:
            builder = tllm.Builder()
            net = builder.create_network()
            net.plugin_config.set_nccl_plugin(dtype)
            init_all_reduce_helper()
            _buffers, workspace = current_all_reduce_helper(
            ).allocate_workspace(mapping, size * dtype_size)

            with tllm.net_guard(net):
                tllm.default_trtnet()

                x = Tensor(name='x',
                           shape=input.shape,
                           dtype=tllm.str_dtype_to_trt(dtype))

                current_all_reduce_helper().set_workspace_tensor(mapping)

                current = x
                for _ in range(inner_loop):
                    current = allreduce(
                        current,
                        mapping.tp_group,
                        all_reduce_params=AllReduceParams(strategy=strategy))
                current.mark_output('output', dtype)
            feed_dict = {'x': input, 'all_reduce_workspace': workspace}
            builder_config = builder.create_builder_config(precision=dtype)
            engine = builder.build_engine(net, builder_config)
            assert engine is not None, "Failed to build engine"
            session = Session.from_serialized_engine(engine)

            _, start = cuda.cuEventCreate(0)
            _, stop = cuda.cuEventCreate(0)
            runtimes = []

            tllm.mpi_barrier()
            output = torch.empty(input.shape, dtype=torch_dtype, device='cuda')
            stream = torch.cuda.current_stream()
            for _ in range(10):
                cuda.cuEventRecord(start, stream.cuda_stream)
                session.run(inputs=feed_dict,
                            outputs={"output": output},
                            stream=stream.cuda_stream)
                cuda.cuEventRecord(stop, stream.cuda_stream)
                torch.cuda.synchronize()
                _, ms = cuda.cuEventElapsedTime(start, stop)
                runtimes.append(ms)

            median_ms = sorted(runtimes)[len(runtimes) // 2]

            allreduce_ref = (input * world_size)**inner_loop
            assert torch.allclose(output, allreduce_ref)

            if mapping.rank == 0:
                print(
                    f"{mapping.world_size:<15}, {dtype:<10}, {size:<15}, {strategy.name:<15}, {median_ms:<10.2f}"
                )

        size *= ratio


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dtype", "-t", default="float16")
    parser.add_argument(
        "--range",
        "-r",
        default="256,256000000,10",  # 256 to 256M
        help="min_size,max_size,multiplicative_ratio")
    parser.add_argument("--no-header", action="store_true")
    args = parser.parse_args()

    allreduce_benchmark(args.dtype, args.range, args.no_header)
