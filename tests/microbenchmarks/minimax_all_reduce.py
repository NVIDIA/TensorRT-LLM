# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from tensorrt_llm._torch.distributed import MiniMaxAllReduceRMS
from tensorrt_llm._utils import local_mpi_rank, local_mpi_size, nvtx_range
from tensorrt_llm.bindings.internal.runtime import delay_kernel
from tensorrt_llm.logger import logger
from tensorrt_llm.plugin.plugin import CustomAllReduceHelper

# MiniMax all-reduce only uses D (hidden_size) 128 and 1536 in practice.
ALLOWED_HIDDEN_SIZES = (128, 1536)


def profile_minimax_allreduce_rms(
    mapping: Mapping,
    op: MiniMaxAllReduceRMS,
    enable_cudagraph: bool = False,
    inner_loop: int = 200,
    outer_loop: int = 10,
    input_tensor=None,
    norm_weight=None,
    eps: float = 1e-5,
):
    def func(loop_num=inner_loop):
        out = None
        for _ in range(loop_num):
            out = op(input_tensor, norm_weight, eps)
        return out

    start = [torch.cuda.Event(enable_timing=True) for _ in range(outer_loop)]
    stop = [torch.cuda.Event(enable_timing=True) for _ in range(outer_loop)]
    graph = torch.cuda.CUDAGraph()

    stream = torch.cuda.Stream()
    with (
        torch.cuda.stream(stream),
        nvtx_range(f"minimax_allreduce_rms: shape={input_tensor.size(0)}x{input_tensor.size(1)}"),
    ):
        func(loop_num=1)

        if enable_cudagraph:
            for i in range(2):
                func(loop_num=1)
            with torch.cuda.graph(graph, stream=stream):
                _ = func()

        delay_kernel(20000, stream)

        torch.cuda.synchronize()
        torch.cuda.profiler.start()

        for i in range(outer_loop):
            start[i].record(stream)
            if enable_cudagraph:
                graph.replay()
            else:
                _ = func()
            stop[i].record(stream)

    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    runtimes = [start[i].elapsed_time(stop[i]) for i in range(outer_loop)]
    median_ms = sorted(runtimes)[len(runtimes) // 2] / inner_loop
    return median_ms


def minimax_allreduce_benchmark(
    dtype: str = "bfloat16",
    test_range: str = "256,256000000,10",
    enable_cudagraph: bool = False,
    explore_2d: bool = False,
    save_csv: str = None,
):
    world_size = tllm.mpi_world_size()
    rank = tllm.mpi_rank()
    local_rank = local_mpi_rank()
    gpus_per_node = local_mpi_size()

    torch.cuda.set_device(local_rank)
    cudart.cudaSetDevice(local_rank)

    mapping = Mapping(world_size, rank, gpus_per_node, tp_size=world_size)
    logger.set_rank(mapping.rank)

    if world_size == 1:
        raise RuntimeError("Benchmark must run with mpi_world_size > 1")

    torch_dtype = tllm._utils.str_dtype_to_torch(dtype)

    inner_loop = 200
    outer_loop = 10
    eps = 1e-5

    shape_list = []
    if explore_2d:
        num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        hidden_size_list = list(ALLOWED_HIDDEN_SIZES)
        for num_tokens, hidden_size in product(num_tokens_list, hidden_size_list):
            shape_list.append((num_tokens, hidden_size))
    else:
        min_size, max_size, ratio = [int(i) for i in test_range.split(",")]
        for hidden_size in ALLOWED_HIDDEN_SIZES:
            n_min = max(1, (min_size + hidden_size - 1) // hidden_size)
            n_max = max_size // hidden_size
            num_tokens = n_min
            while num_tokens <= n_max:
                shape_list.append((num_tokens, hidden_size))
                num_tokens *= ratio
                num_tokens = max(num_tokens, 1)
    # Only test D (hidden_size) = 128 and 1536 (no-op when explore_2d already uses them)
    shape_list = [(n, d) for n, d in shape_list if d in ALLOWED_HIDDEN_SIZES]

    op = MiniMaxAllReduceRMS(mapping=mapping)
    max_workspace = CustomAllReduceHelper.max_workspace_size_auto(
        mapping.tp_size, support_deterministic=False
    )

    df = pd.DataFrame()
    for num_tokens, hidden_size in shape_list:
        message_size_bytes = num_tokens * hidden_size * torch.finfo(torch_dtype).bits // 8
        if message_size_bytes > max_workspace:
            continue

        input_tensor = torch.ones((num_tokens, hidden_size), dtype=torch_dtype, device="cuda")
        norm_weight = torch.randn((hidden_size,), dtype=torch_dtype, device="cuda")

        median_ms = profile_minimax_allreduce_rms(
            mapping=mapping,
            op=op,
            enable_cudagraph=enable_cudagraph,
            inner_loop=inner_loop,
            outer_loop=outer_loop,
            input_tensor=input_tensor,
            norm_weight=norm_weight,
            eps=eps,
        )

        if mapping.rank == 0:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "world_size": [mapping.world_size],
                            "dtype": [dtype],
                            "message_size_bytes": [message_size_bytes],
                            "num_tokens": [num_tokens],
                            "hidden_size": [hidden_size],
                            "time (us)": [median_ms * 1000],
                        }
                    ),
                ]
            )
            print(
                f"num_tokens: {num_tokens}, hidden_size: {hidden_size}, "
                f"time (us): {median_ms * 1000}"
            )

    if mapping.rank == 0:
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)
        print(df)

    if mapping.rank == 0 and save_csv is not None:
        df.to_csv(save_csv, index=False)

    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dtype", "-t", default="bfloat16")
    parser.add_argument(
        "--range",
        "-r",
        default="256,256000000,10",
        help="min_size,max_size,multiplicative_ratio",
    )
    parser.add_argument("--explore_2d", action="store_true", default=False)
    parser.add_argument("--enable_cudagraph", action="store_true")
    parser.add_argument("--save_csv", type=str, default=None)

    args = parser.parse_args()

    minimax_allreduce_benchmark(
        args.dtype,
        args.range,
        args.enable_cudagraph,
        args.explore_2d,
        args.save_csv,
    )
