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
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

import tensorrt_llm as tllm
from tensorrt_llm import Mapping
from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._utils import local_mpi_rank, local_mpi_size
from tensorrt_llm.bindings.internal.runtime import delay_kernel
from tensorrt_llm.functional import AllReduceParams, AllReduceStrategy


def allreduce_benchmark(dtype: str,
                        test_range: str = "1,10000000,10",
                        no_header: bool = False,
                        enable_cudagraph: bool = False):
    tllm.logger.set_level('error')
    world_size = tllm.mpi_world_size()
    rank = tllm.mpi_rank()
    local_rank = local_mpi_rank()
    gpus_per_node = local_mpi_size()

    torch.cuda.set_device(local_rank)
    cudart.cudaSetDevice(local_rank)

    mapping = Mapping(world_size, rank, gpus_per_node, tp_size=world_size)

    if world_size == 1:
        raise RuntimeError("Benchmark must run with mpi_world_size > 1")

    torch_dtype = tllm._utils.str_dtype_to_torch(dtype)
    min_size, max_size, ratio = [int(i) for i in test_range.split(",")]
    inner_loop = 1200
    outer_loop = 10

    size = min_size
    hidden_size = size
    bs = 1
    if mapping.rank == 0 and not no_header:
        print(
            f"{'world_size':<15}, {'dtype':<10}, {'message size':<15}, {'strategy':<10}, {'fusion':<20}, {'version':<10}, {'duration (ms)':<10}"
        )
    while size < max_size:
        input = torch.ones((bs, hidden_size), dtype=torch_dtype, device="cuda")

        for version in ["v1"]:
            for fusion in [
                    AllReduceFusionOp.RESIDUAL_RMS_NORM, AllReduceFusionOp.NONE
            ]:
                for strategy in [
                        AllReduceStrategy.NCCL,
                        AllReduceStrategy.ONESHOT,
                        AllReduceStrategy.TWOSHOT,
                ]:
                    if size >= 25600000 and fusion != AllReduceFusionOp.NONE:
                        continue
                    allreduce = AllReduce(mapping=mapping, strategy=strategy)
                    if fusion == AllReduceFusionOp.RESIDUAL_RMS_NORM:
                        norm_weight = torch.randn((hidden_size, ),
                                                  dtype=torch_dtype,
                                                  device="cuda")
                        norm = RMSNorm(hidden_size=hidden_size,
                                       dtype=torch_dtype,
                                       eps=1e-5).cuda()
                        norm.weight.data.copy_(norm_weight)
                        if version == "v1":
                            params = {
                                "all_reduce_params":
                                AllReduceParams(fusion_op=fusion,
                                                residual=input,
                                                norm_weight=norm.weight,
                                                eps=norm.variance_epsilon)
                            }
                        else:
                            params = {
                                "reduce_fusion_inputs": [input, norm.weight],
                                "eps": norm.variance_epsilon,
                                "fusion_op": fusion
                            }
                    else:
                        if version == "v1":
                            params = {
                                "all_reduce_params":
                                AllReduceParams(fusion_op=fusion)
                            }
                        else:
                            continue

                    def func(input):
                        for _ in range(inner_loop):
                            input = allreduce(input, **params)
                            if fusion == AllReduceFusionOp.RESIDUAL_RMS_NORM:
                                input = input[0]
                        return input

                    start = [
                        torch.cuda.Event(enable_timing=True)
                        for _ in range(outer_loop)
                    ]
                    stop = [
                        torch.cuda.Event(enable_timing=True)
                        for _ in range(outer_loop)
                    ]
                    graph = torch.cuda.CUDAGraph()

                    stream = torch.cuda.Stream()
                    with torch.cuda.stream(stream):
                        if enable_cudagraph:
                            for _ in range(2):
                                func(input)
                            with torch.cuda.graph(graph, stream=stream):
                                output = func(input)
                        tllm.mpi_barrier()
                        delay_kernel(2000000, stream)
                        torch.cuda.profiler.start()
                        for i in range(outer_loop):
                            start[i].record(stream)
                            if enable_cudagraph:
                                graph.replay()
                            else:
                                output = func(input)
                            stop[i].record(stream)

                    torch.cuda.synchronize()
                    torch.cuda.profiler.stop()
                    runtimes = [
                        start[i].elapsed_time(stop[i])
                        for i in range(outer_loop)
                    ]
                    median_ms = sorted(runtimes)[len(runtimes) // 2]

                    if fusion == AllReduceFusionOp.NONE:
                        allreduce_ref = (input * world_size)**inner_loop
                        torch.testing.assert_close(output, allreduce_ref)

                    if mapping.rank == 0:
                        print(
                            f"{mapping.world_size:<15}, {dtype:<10}, {size:<15}, {strategy.name:<10}, {fusion.name:<20}, {version:<10}, {median_ms:<10.2f}"
                        )

        size *= ratio
        if hidden_size * ratio > 4096:
            bs *= ratio
        else:
            hidden_size *= ratio
        assert size == bs * hidden_size


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dtype", "-t", default="float16")
    parser.add_argument(
        "--range",
        "-r",
        default="256,256000000,10",  # 256 to 256M
        help="min_size,max_size,multiplicative_ratio")
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--enable-cudagraph", action="store_true")
    args = parser.parse_args()

    allreduce_benchmark(args.dtype, args.range, args.no_header,
                        args.enable_cudagraph)
