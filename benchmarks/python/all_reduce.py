# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import tensorrt as trt
# isort: on
from cuda import cuda, cudart
from mpi4py import MPI
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork

import tensorrt_llm as tllm
from tensorrt_llm import Mapping, Tensor
from tensorrt_llm._ipc_utils import IpcMemory, peer_access
from tensorrt_llm.functional import AllReduceStrategy, allreduce


def allreduce_benchmark(dtype: str, test_range: str = "10,10000000,10"):
    tllm.logger.set_level('error')
    world_size = tllm.mpi_world_size()
    rank = tllm.mpi_rank()

    torch.cuda.set_device(rank)
    cudart.cudaSetDevice(rank)

    mapping = Mapping(world_size, rank, world_size, world_size)

    if world_size == 1:
        raise RuntimeError("Benchmark must run with mpi_world_size > 1")

    ipc_barriers_in = IpcMemory(
        mapping, IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * mapping.tp_size)
    ipc_barriers_out = IpcMemory(
        mapping, IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * mapping.tp_size)
    torch_dtype = tllm._utils.str_dtype_to_torch(dtype)

    min_size, max_size, ratio = [int(i) for i in test_range.split(",")]
    inner_loop = 1000

    size = min_size
    while size < max_size:
        ipc_buffers = IpcMemory(mapping, size * 4)
        workspace = torch.tensor(ipc_buffers.serialize() +
                                 ipc_barriers_in.serialize() +
                                 ipc_barriers_out.serialize(),
                                 dtype=torch.int64,
                                 device="cpu")

        input = torch.zeros(size, dtype=torch_dtype, device="cuda")

        for strategy in [
                AllReduceStrategy.RING, AllReduceStrategy.ONESHOT,
                AllReduceStrategy.TWOSHOT
        ]:
            builder = tllm.Builder()
            net = builder.create_network()
            net.plugin_config.set_nccl_plugin(dtype)

            with tllm.net_guard(net):
                network = tllm.default_trtnet()

                x = Tensor(name='x',
                           shape=input.shape,
                           dtype=tllm.str_dtype_to_trt(dtype))

                w = Tensor(name='workspace',
                           shape=workspace.shape,
                           dtype=trt.int64)

                current = x
                for i in range(inner_loop):
                    current = allreduce(
                        current, mapping.tp_group,
                        w if strategy != AllReduceStrategy.RING else None, i,
                        strategy)
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
            feed_dict = {'x': input, 'workspace': workspace}

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
