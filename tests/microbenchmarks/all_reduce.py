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
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             Distributed)
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._utils import (get_sm_version, local_mpi_rank, local_mpi_size,
                                 nvtx_range)
from tensorrt_llm.bindings.internal.runtime import delay_kernel
from tensorrt_llm.functional import AllReduceParams, AllReduceStrategy
from tensorrt_llm.logger import logger
from tensorrt_llm.plugin.plugin import CustomAllReduceHelper


def profile_allreduce(
    mapping: Mapping,
    dist: Distributed,
    enable_cudagraph: bool = False,
    inner_loop=200,
    outer_loop=10,
    strategy=AllReduceStrategy.NCCL,
    fusion=AllReduceFusionOp.NONE,
    input=None,
    residual=None,
    norm=None,
    scale=None,
    bias=None,
    profile_gemm_allreduce: bool = False,
    gemm_in_features: int | None = None,
):

    allreduce_params = AllReduceParams(
        fusion_op=fusion,
        residual=residual,
        norm_weight=norm.weight,
        eps=norm.variance_epsilon,
        scale=scale,
        bias=bias,
    )

    allreduce = AllReduce(mapping=mapping, strategy=strategy)
    linear = None
    if profile_gemm_allreduce:
        if gemm_in_features is None:
            raise ValueError(
                "gemm_in_features must be provided when profile_gemm_allreduce is enabled"
            )
        linear = Linear(
            in_features=gemm_in_features,
            out_features=gemm_in_features,
            bias=False,
            dtype=input.dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            reduce_output=True,
            allreduce_strategy=strategy,
        ).to(input.device)
        torch.nn.init.normal_(linear.weight, mean=0.0, std=0.02)

    def func(x, loop_num=inner_loop):
        for _ in range(loop_num):
            if profile_gemm_allreduce:
                output = linear(x, all_reduce_params=allreduce_params)
            else:
                output = allreduce(x, all_reduce_params=allreduce_params)
        return output if fusion == AllReduceFusionOp.NONE else output[0]

    start = [torch.cuda.Event(enable_timing=True) for _ in range(outer_loop)]
    stop = [torch.cuda.Event(enable_timing=True) for _ in range(outer_loop)]
    graph = torch.cuda.CUDAGraph()

    stream = torch.cuda.Stream()
    shape_hidden = gemm_in_features if profile_gemm_allreduce else input.size(1)
    with torch.cuda.stream(stream), nvtx_range(
            f"allreudce: shape={input.size(0)}x{shape_hidden} fusion={fusion} "
            f"strategy={strategy} mode={'gemm_ar' if profile_gemm_allreduce else 'allreduce'}"
    ):
        with autotune():
            func(input, loop_num=1)

        if enable_cudagraph:
            # Untimed warmup run outside of graph capture
            func(input, loop_num=1)
            # CUDA graph warmup then capture
            for _ in range(2):
                func(input, loop_num=1)
            with torch.cuda.graph(graph, stream=stream):
                output = func(input)

        dist.barrier()
        # add delay to avoid the effect of host time overhead
        delay_kernel(20000, stream)

        torch.cuda.synchronize()
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
    runtimes = [start[i].elapsed_time(stop[i]) for i in range(outer_loop)]
    median_ms = sorted(runtimes)[len(runtimes) // 2] / inner_loop

    if fusion == AllReduceFusionOp.NONE and not profile_gemm_allreduce:
        allreduce_ref = (input * mapping.world_size)
        torch.testing.assert_close(
            output,
            allreduce_ref,
            atol=1e-2,
            rtol=1e-2,
            msg="Allreduce result mismatch",
        )
    return median_ms


def allreduce_benchmark(
    dtype: str = 'bfloat16',
    test_range: str = "256,256000000,10",
    enable_cudagraph: bool = False,
    explore_2d: bool = False,
    save_csv: str = None,
    enable_auto: bool = False,
    profile_gemm_allreduce: bool = False,
):
    world_size = tllm.mpi_world_size()
    rank = tllm.mpi_rank()
    local_rank = local_mpi_rank()
    gpus_per_node = local_mpi_size()

    torch.cuda.set_device(local_rank)
    cudart.cudaSetDevice(local_rank)

    mapping = Mapping(world_size, rank, gpus_per_node, tp_size=world_size)

    logger.set_rank(mapping.rank)

    AutoTuner.get().setup_distributed_state(mapping)
    dist = Distributed.get(mapping)

    sm_version = get_sm_version()

    if world_size == 1:
        raise RuntimeError("Benchmark must run with mpi_world_size > 1")

    torch_dtype = tllm._utils.str_dtype_to_torch(dtype)

    inner_loop = 200
    outer_loop = 10

    # generate shape list
    shape_list = []

    if explore_2d:
        num_tokens_list = [
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
        ]
        hidden_size_list = [128, 256, 512, 1024, 2048, 4096, 8192]
        for num_tokens, hidden_size in product(num_tokens_list,
                                               hidden_size_list):
            shape_list.append((num_tokens, hidden_size))
    else:
        min_size, max_size, ratio = [int(i) for i in test_range.split(",")]
        size = min_size
        hidden_size = min_size
        num_tokens = 1
        while size < max_size:
            size *= ratio
            shape_list.append((num_tokens, hidden_size))
            if hidden_size * ratio > 4096:
                num_tokens *= ratio
            else:
                hidden_size *= ratio
            assert size == num_tokens * hidden_size

    fusion_patterns = [
        AllReduceFusionOp.NONE,
        AllReduceFusionOp.RESIDUAL_RMS_NORM,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
    ]
    strategies = [
        AllReduceStrategy.NCCL,
        AllReduceStrategy.NCCL_SYMMETRIC,
        AllReduceStrategy.ONESHOT,
        AllReduceStrategy.TWOSHOT,
        AllReduceStrategy.AUTO,
    ]
    df = pd.DataFrame()
    for (num_tokens, hidden_size) in shape_list:
        message_size = num_tokens * hidden_size * torch.finfo(
            torch_dtype).bits // 8

        if profile_gemm_allreduce and hidden_size % mapping.tp_size != 0:
            continue

        if message_size > CustomAllReduceHelper.max_workspace_size_auto(
                mapping.tp_size):
            continue

        input = torch.ones((num_tokens, hidden_size),
                           dtype=torch_dtype,
                           device="cuda")
        residual = torch.randn_like(input)
        norm_weight = torch.randn((hidden_size, ),
                                  dtype=torch_dtype,
                                  device="cuda")
        norm = RMSNorm(hidden_size=hidden_size, dtype=torch_dtype,
                       eps=1e-5).cuda()
        norm.weight.data.copy_(norm_weight)
        scale = torch.tensor(1.0, dtype=torch.float32).cuda()

        for fusion, strategy in product(fusion_patterns, strategies):
            if num_tokens < mapping.tp_size and strategy == AllReduceStrategy.TWOSHOT:
                continue

            if fusion == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4 and sm_version < 100:
                continue

            if not enable_auto and strategy == AllReduceStrategy.AUTO:
                continue

            input_for_profile = input
            if profile_gemm_allreduce:
                local_in_features = hidden_size // mapping.tp_size
                start_col = mapping.tp_rank * local_in_features
                input_for_profile = input[:, start_col:start_col +
                                          local_in_features].contiguous()
            elif strategy == AllReduceStrategy.NCCL_SYMMETRIC:
                try:
                    window_out, is_valid = torch.ops.trtllm.create_nccl_window_tensor(
                        mapping.tp_group, list(input.shape), input.dtype)
                except Exception:
                    window_out, is_valid = None, False
                if bool(is_valid) and window_out is not None:
                    window_out.copy_(input)
                    input_for_profile = window_out

            median_ms = profile_allreduce(
                mapping=mapping,
                dist=dist,
                enable_cudagraph=enable_cudagraph,
                inner_loop=inner_loop,
                outer_loop=outer_loop,
                strategy=strategy,
                fusion=fusion,
                input=input_for_profile,
                residual=residual,
                norm=norm,
                scale=scale,
                profile_gemm_allreduce=profile_gemm_allreduce,
                gemm_in_features=hidden_size,
            )

            if mapping.rank == 0:
                df = pd.concat([
                    df,
                    pd.DataFrame({
                        "world_size": [mapping.world_size],
                        "dtype": [dtype],
                        "size": [message_size],
                        "num_tokens": [num_tokens],
                        "hidden_size": [hidden_size],
                        "strategy": [strategy.name],
                        "fusion": [fusion.name],
                        "op": [
                            "gemm_allreduce"
                            if profile_gemm_allreduce else "allreduce"
                        ],
                        "time (us)": [median_ms * 1000]
                    })
                ])

                # print the new record in a single line instead of a dataframe
                if mapping.rank == 0:
                    print(
                        f"num_tokens: {num_tokens}, hidden_size: {hidden_size}, "
                        f"strategy: {strategy.name}, fusion: {fusion.name}, "
                        f"op: {'gemm_allreduce' if profile_gemm_allreduce else 'allreduce'}, "
                        f"time (us): {median_ms * 1000}")

    AutoTuner.get().print_profiling_cache()
    # print the dataframe
    if mapping.rank == 0:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(df)

    # # save the dataframe to a csv file
    if mapping.rank == 0 and save_csv is not None:
        df.to_csv(f"{save_csv}", index=False)

    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dtype", "-t", default="bfloat16")
    parser.add_argument(
        "--range",
        "-r",
        default="256,256000000,10",  # 256 to 256M
        help="min_size,max_size,multiplicative_ratio")
    parser.add_argument("--explore_2d", action="store_true", default=False)
    parser.add_argument("--enable_cudagraph", action="store_true")
    parser.add_argument("--save_csv", type=str, default=None)
    parser.add_argument("--enable_auto", action="store_true", default=False)
    parser.add_argument("--profile_gemm_allreduce",
                        action="store_true",
                        default=False)

    args = parser.parse_args()

    allreduce_benchmark(
        args.dtype,
        args.range,
        args.enable_cudagraph,
        args.explore_2d,
        args.save_csv,
        args.enable_auto,
        args.profile_gemm_allreduce,
    )
