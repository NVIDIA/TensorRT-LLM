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
from itertools import product

# isort: off
import torch
import pandas as pd
# isort: on
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

import tensorrt_llm as tllm
from tensorrt_llm import Mapping
from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._utils import local_mpi_rank, local_mpi_size, nvtx_range
from tensorrt_llm.bindings.internal.runtime import delay_kernel
from tensorrt_llm.functional import AllReduceParams, AllReduceStrategy
from tensorrt_llm.plugin.plugin import CustomAllReduceHelper


def allreduce_benchmark(dtype: str,
                        test_range: str = "1,10000000,10",
                        no_header: bool = False,
                        enable_cudagraph: bool = False,
                        output_csv: str = None):
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
    inner_loop = 200
    outer_loop = 10

    hidden_size = min_size
    num_tokens = 8
    size = max(num_tokens * hidden_size, min_size)

    # generate shape list
    shape_list = []
    # make it simple
    while size < max_size:
        shape_list.append((num_tokens, hidden_size))
        size *= ratio
        if hidden_size * ratio > 4096:
            num_tokens *= ratio
        else:
            hidden_size *= ratio

    fusion_patterns = [
        AllReduceFusionOp.NONE,
        AllReduceFusionOp.RESIDUAL_RMS_NORM,
    ]
    strategies = [
        AllReduceStrategy.NCCL,
        AllReduceStrategy.ONESHOT,
        AllReduceStrategy.TWOSHOT,
    ]
    versions = ["v1"]
    df = pd.DataFrame()
    for (num_tokens, hidden_size) in shape_list:
        message_size = num_tokens * hidden_size * torch.finfo(
            torch_dtype).bits // 8
        if num_tokens < mapping.tp_size:
            continue
        if message_size > CustomAllReduceHelper.max_workspace_size_auto(
                mapping.tp_size):
            continue

        input = torch.ones((num_tokens, hidden_size),
                           dtype=torch_dtype,
                           device="cuda")
        norm_weight = torch.randn((hidden_size, ),
                                  dtype=torch_dtype,
                                  device="cuda")
        norm = RMSNorm(hidden_size=hidden_size, dtype=torch_dtype,
                       eps=1e-5).cuda()
        norm.weight.data.copy_(norm_weight)
        for version, fusion, strategy in product(versions, fusion_patterns,
                                                 strategies):
            allreduce = AllReduce(mapping=mapping, strategy=strategy)
            allreduce_params = AllReduceParams(
                fusion_op=fusion,
                residual=input,
                norm_weight=norm.weight,
                eps=norm.variance_epsilon,
            )

            allreduce = AllReduce(mapping=mapping, strategy=strategy)

            def func(input):
                for _ in range(inner_loop):
                    output = allreduce(input,
                                       all_reduce_params=allreduce_params)
                return output if fusion == AllReduceFusionOp.NONE else output[0]

            start = [
                torch.cuda.Event(enable_timing=True) for _ in range(outer_loop)
            ]
            stop = [
                torch.cuda.Event(enable_timing=True) for _ in range(outer_loop)
            ]
            graph = torch.cuda.CUDAGraph()

            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream), nvtx_range(
                    f"allreudce: shape={num_tokens}x{hidden_size} fusion={fusion} strategy={strategy}"
            ):
                if enable_cudagraph:
                    # CUDA graph warmup then capture
                    for _ in range(2):
                        func(input)
                    with torch.cuda.graph(graph, stream=stream):
                        output = func(input)
                # warmup for no cuda graph
                func(input)

                tllm.mpi_barrier()
                # add delay to avoid the effect of host time overhead
                delay_kernel(20000, stream)

                torch.cuda.synchronize()
                torch.cuda.profiler.start()

                # start profiling loop
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
                start[i].elapsed_time(stop[i]) for i in range(outer_loop)
            ]
            median_ms = sorted(runtimes)[len(runtimes) // 2] / inner_loop

            if fusion == AllReduceFusionOp.NONE:
                allreduce_ref = (input * world_size)
                torch.testing.assert_close(output,
                                           allreduce_ref,
                                           atol=1e-2,
                                           rtol=1e-2,
                                           msg="Allreduce result mismatch")

            if mapping.rank == 0:
                df = pd.concat([
                    df,
                    pd.DataFrame({
                        "world_size": [mapping.world_size],
                        "dtype": [dtype],
                        "size": [message_size],
                        "shape": [f"{num_tokens}x{hidden_size}"],
                        "strategy": [strategy.name],
                        "fusion": [fusion.name],
                        "version": [version],
                        "time (us)": [median_ms * 1000]
                    })
                ])

    # print the dataframe
    if mapping.rank == 0:
        print(df)

    # # save the dataframe to a csv file
    if mapping.rank == 0:
        df.to_csv(f"{output_csv}", index=False)


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
    parser.add_argument("--output_csv", type=str, default=None)

    args = parser.parse_args()

    allreduce_benchmark(args.dtype, args.range, args.no_header,
                        args.enable_cudagraph, args.output_csv)
