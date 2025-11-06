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
import tensorrt_llm.bindings.internal.userbuffers as ub
from tensorrt_llm import Mapping
from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._utils import (get_sm_version, local_mpi_rank, local_mpi_size,
                                 nvtx_range)
from tensorrt_llm.bindings.internal.runtime import delay_kernel
from tensorrt_llm.functional import AllReduceParams, AllReduceStrategy
from tensorrt_llm.plugin.plugin import CustomAllReduceHelper


def profile_allreduce(
    mapping: Mapping,
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

    def func(x):
        for _ in range(inner_loop):
            output = allreduce(x, all_reduce_params=allreduce_params)
        return output if fusion == AllReduceFusionOp.NONE else output[0]

    start = [torch.cuda.Event(enable_timing=True) for _ in range(outer_loop)]
    stop = [torch.cuda.Event(enable_timing=True) for _ in range(outer_loop)]
    graph = torch.cuda.CUDAGraph()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), nvtx_range(
            f"allreudce: shape={input.size(0)}x{input.size(1)} fusion={fusion} strategy={strategy}"
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

    if fusion == AllReduceFusionOp.NONE:
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
    strategy: str = None,
    inner_loop: int = 200,
    outer_loop: int = 10,
    tokens_range: str = "1,16384,2",
    hidden_sizes_range: str = "128,8192,2",
):
    """
    Benchmark AllReduce operations.

    Args:
        dtype: Data type for benchmarking
        test_range: Range specification (min,max,ratio)
        enable_cudagraph: Enable CUDA graph capture
        explore_2d: Explore 2D parameter space (num_tokens x hidden_size)
        save_csv: Path to save CSV results
        strategy: Specific strategy to test (if None, tests default set: NCCL, NCCL_SYMMETRIC, NCCL_DEVICE, MNNVL)
        inner_loop: Number of iterations per timing measurement (default: 200)
        outer_loop: Number of timing measurements to take (default: 10)
        tokens_range: Range for number of tokens in 2D mode (min,max,ratio) (default: "1,16384,2")
        hidden_sizes_range: Range for hidden sizes in 2D mode (min,max,ratio) (default: "128,8192,2")
    """
    world_size = tllm.mpi_world_size()
    rank = tllm.mpi_rank()
    local_rank = local_mpi_rank()
    gpus_per_node = local_mpi_size()

    if world_size == 1:
        if rank == 0:
            print("ERROR: Benchmark must run with mpi_world_size > 1",
                  file=sys.stderr,
                  flush=True)
        sys.exit(1)

    # Device setup
    torch.cuda.set_device(local_rank)
    cudart.cudaSetDevice(local_rank)

    mapping = Mapping(world_size, rank, gpus_per_node, tp_size=world_size)
    sm_version = get_sm_version()

    # Data type setup
    torch_dtype = tllm._utils.str_dtype_to_torch(dtype)
    dtype_size_bytes = torch_dtype.itemsize

    # Parse test range
    min_size, max_size, ratio = [int(i) for i in test_range.split(",")]

    # generate shape list
    shape_list = []

    if explore_2d:
        # Parse tokens range
        min_tokens, max_tokens, tokens_ratio = [
            int(i) for i in tokens_range.split(",")
        ]

        # Parse hidden sizes range
        min_hidden, max_hidden, hidden_ratio = [
            int(i) for i in hidden_sizes_range.split(",")
        ]

        # Generate token counts list
        num_seqs_list = []
        current = min_tokens
        while current <= max_tokens:
            num_seqs_list.append(current)
            current *= tokens_ratio

        # Generate hidden sizes list
        hidden_size_list = []
        current = min_hidden
        while current <= max_hidden:
            hidden_size_list.append(current)
            current *= hidden_ratio

        # Create all combinations
        for num_tokens, hidden_size in product(num_seqs_list, hidden_size_list):
            shape_list.append((num_tokens, hidden_size))
    else:
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

    # Map strategy names to enum values
    strategy_map = {
        "NCCL": AllReduceStrategy.NCCL,
        "MIN_LATENCY": AllReduceStrategy.MIN_LATENCY,
        "NCCL_SYMMETRIC": AllReduceStrategy.NCCL_SYMMETRIC,
        "NCCL_DEVICE": AllReduceStrategy.NCCL_DEVICE,
        "MNNVL": AllReduceStrategy.MNNVL,
        "UB": AllReduceStrategy.UB,
        "ONESHOT": AllReduceStrategy.ONESHOT,
        "TWOSHOT": AllReduceStrategy.TWOSHOT,
        "AUTO": AllReduceStrategy.AUTO,
    }

    # Select strategies based on input
    if strategy:
        # Single strategy specified
        if strategy.upper() not in strategy_map:
            raise ValueError(
                f"Unknown strategy: {strategy}. Available: {', '.join(strategy_map.keys())}"
            )
        strategies = [strategy_map[strategy.upper()]]
    else:
        # Default: test main strategies
        strategies = [
            AllReduceStrategy.NCCL,
            AllReduceStrategy.NCCL_SYMMETRIC,
            AllReduceStrategy.NCCL_DEVICE,
            AllReduceStrategy.MNNVL,
        ]

    # Validate strategy compatibility for user buffer initialization
    # NCCL_SYMMETRIC and NCCL_DEVICE need UB with use_multicast=True
    # UB strategy needs UB with use_multicast=False
    # These two groups cannot be mixed in a single run
    ub_multicast_strategies = {
        AllReduceStrategy.NCCL_SYMMETRIC, AllReduceStrategy.NCCL_DEVICE
    }
    ub_no_multicast_strategies = {AllReduceStrategy.UB}

    has_multicast_strategies = any(s in ub_multicast_strategies
                                   for s in strategies)
    has_no_multicast_strategies = any(s in ub_no_multicast_strategies
                                      for s in strategies)

    # Error out if incompatible strategies are mixed
    if has_multicast_strategies and has_no_multicast_strategies:
        multicast_strats = [
            s.name for s in strategies if s in ub_multicast_strategies
        ]
        no_multicast_strats = [
            s.name for s in strategies if s in ub_no_multicast_strategies
        ]
        raise ValueError(
            f"Incompatible strategies selected: {multicast_strats} require use_multicast=True "
            f"while {no_multicast_strats} require use_multicast=False. "
            f"Please run these strategies separately using --strategy.")

    # Initialize user buffers if any strategy needs it
    needs_ub = has_multicast_strategies or has_no_multicast_strategies

    if needs_ub:
        max_bytes = max_size * dtype_size_bytes
        use_multicast = has_multicast_strategies  # True for NCCL_SYMMETRIC/NCCL_DEVICE, False for UB

        ub.initialize_userbuffers_manager(world_size, 1, 1, rank,
                                          torch.cuda.device_count(), max_bytes,
                                          use_multicast)

    df = pd.DataFrame()
    for (num_tokens, hidden_size) in shape_list:
        message_size = num_tokens * hidden_size * torch.finfo(
            torch_dtype).bits // 8

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

            # UB strategy doesn't support NONE fusion
            if strategy == AllReduceStrategy.UB and fusion == AllReduceFusionOp.NONE:
                continue

            median_ms = profile_allreduce(
                mapping=mapping,
                enable_cudagraph=enable_cudagraph,
                inner_loop=inner_loop,
                outer_loop=outer_loop,
                strategy=strategy,
                fusion=fusion,
                input=input,
                residual=residual,
                norm=norm,
                scale=scale,
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
                        "time (us)": [median_ms * 1000]
                    })
                ])

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
    parser.add_argument("--dtype",
                        "-t",
                        default="bfloat16",
                        help="Data type for benchmarking")
    parser.add_argument(
        "--range",
        "-r",
        default="256,256000000,4",  # 256 to 256M
        help="min_size,max_size,multiplicative_ratio")
    parser.add_argument(
        "--explore_2d",
        action="store_true",
        default=False,
        help="Explore 2D parameter space (num_tokens x hidden_size)")
    parser.add_argument("--enable_cudagraph",
                        action="store_true",
                        help="Enable CUDA graph capture")
    parser.add_argument("--save_csv",
                        type=str,
                        default=None,
                        help="Path to save CSV results")
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help=
        "Test specific strategy. If not specified, defaults to: NCCL, NCCL_SYMMETRIC, NCCL_DEVICE, MNNVL. "
        "Available: NCCL, NCCL_SYMMETRIC, NCCL_DEVICE, MNNVL, MIN_LATENCY, UB, ONESHOT, TWOSHOT, AUTO"
    )
    parser.add_argument(
        "--inner_loop",
        type=int,
        default=200,
        help="Number of iterations per timing measurement (default: 200)")
    parser.add_argument(
        "--outer_loop",
        type=int,
        default=10,
        help="Number of timing measurements to take (default: 10)")
    parser.add_argument(
        "--tokens_range",
        type=str,
        default="1,16384,2",
        help=
        "Range for number of tokens in 2D mode: min,max,ratio (default: 1,16384,2)"
    )
    parser.add_argument(
        "--hidden_sizes_range",
        type=str,
        default="128,8192,2",
        help=
        "Range for hidden sizes in 2D mode: min,max,ratio (default: 128,8192,2)"
    )

    args = parser.parse_args()

    allreduce_benchmark(
        dtype=args.dtype,
        test_range=args.range,
        enable_cudagraph=args.enable_cudagraph,
        explore_2d=args.explore_2d,
        save_csv=args.save_csv,
        strategy=args.strategy,
        inner_loop=args.inner_loop,
        outer_loop=args.outer_loop,
        tokens_range=args.tokens_range,
        hidden_sizes_range=args.hidden_sizes_range,
    )
