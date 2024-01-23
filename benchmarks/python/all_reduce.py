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
from mpi4py import MPI
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork

import tensorrt_llm as tllm
from tensorrt_llm import Mapping, Tensor
from tensorrt_llm._ipc_utils import peer_access
from tensorrt_llm.functional import AllReduceStrategy, allreduce
from tensorrt_llm.plugin.plugin import current_all_reduce_helper


def allreduce_benchmark(dtype: str, test_range: str = "10,10000000,10"):
    tllm.logger.set_level('error')
    world_size = tllm.mpi_world_size()
    rank = tllm.mpi_rank()

    torch.cuda.set_device(rank)
    cudart.cudaSetDevice(rank)

    mapping = Mapping(world_size, rank, world_size, world_size)

    if world_size == 1:
        raise RuntimeError("Benchmark must run with mpi_world_size > 1")

    torch_dtype = tllm._utils.str_dtype_to_torch(dtype)
    min_size, max_size, ratio = [int(i) for i in test_range.split(",")]
    inner_loop = 1000

    size = min_size
    dtype_size = torch.finfo(torch_dtype).bits // 8
    while size < max_size:
        input = torch.ones(size, dtype=torch_dtype, device="cuda")

        for strategy in [
                AllReduceStrategy.RING, AllReduceStrategy.ONESHOT,
                AllReduceStrategy.TWOSHOT
        ]:
            builder = tllm.Builder()
            net = builder.create_network()
            net.plugin_config.set_nccl_plugin(dtype, use_custom_all_reduce=True)
            _buffers, workspace = current_all_reduce_helper(
            ).allocate_workspace(mapping, size * dtype_size)

            with tllm.net_guard(net):
                network = tllm.default_trtnet()

                x = Tensor(name='x',
                           shape=input.shape,
                           dtype=tllm.str_dtype_to_trt(dtype))

                current_all_reduce_helper().set_workspace_tensor(mapping)

                current = x
                for _ in range(inner_loop):
                    current = allreduce(current, mapping.tp_group, strategy)
                output = current.trt_tensor

                output.name = 'output'
                output.dtype = tllm.str_dtype_to_trt(dtype)
                network.mark_output(output)

            build_engine = EngineFromNetwork(
                (builder.trt_builder, net.trt_network),
                config=CreateConfig(
                    fp16=(dtype == 'float16'),
                    bf16=(dtype == 'bfloat16'),
                    precision_constraints='obey',
                ))

            output = torch.zeros_like(input)

            stream = torch.cuda.current_stream()
            feed_dict = {'x': input, 'all_reduce_workspace': workspace}

            session = tllm.runtime.Session.from_engine(build_engine())
            _, start = cuda.cuEventCreate(0)
            _, stop = cuda.cuEventCreate(0)
            with peer_access(mapping):
                MPI.COMM_WORLD.barrier()

                cuda.cuEventRecord(start, stream.cuda_stream)
                session.run(inputs=feed_dict,
                            outputs={"output": output},
                            stream=stream.cuda_stream)
                cuda.cuEventRecord(stop, stream.cuda_stream)
            torch.cuda.synchronize()
            _, ms = cuda.cuEventElapsedTime(start, stop)
            assert torch.allclose(output, (input * world_size)**inner_loop)

            if mapping.rank == 0:
                print(f"{size=}, {strategy=}, {ms=}")

        size *= ratio
        if mapping.rank == 0:
            print("")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dtype", "-t", default="float16")
    parser.add_argument("--range",
                        "-r",
                        default="256,25600000,10",
                        help="min_size,max_size,multiplicative_ratio")
    args = parser.parse_args()

    allreduce_benchmark(args.dtype, args.range)
