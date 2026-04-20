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
from tensorrt_llm._utils import local_mpi_rank, local_mpi_size, mpi_barrier
from tensorrt_llm.logger import logger
from tensorrt_llm.plugin.plugin import CustomAllReduceHelper

# MiniMax all-reduce only uses D (hidden_size) 128 and 1536 in practice.
ALLOWED_HIDDEN_SIZES = (256, 1536)

# Q+K fused API benchmark dimensions
QK_Q_DIM = 1536
QK_K_DIM = 256


def profile_minimax_allreduce_rms(
    mapping: Mapping,
    op: MiniMaxAllReduceRMS,
    warmup: int = 10,
    iters: int = 100,
    inner_loop: int = 8,
    input_tensor=None,
    norm_weight=None,
    eps: float = 1e-5,
):
    def func():
        for _ in range(inner_loop):
            op(input_tensor, norm_weight, eps)

    for _ in range(warmup):
        for i in range(inner_loop):
            op(input_tensor, norm_weight, eps)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        func()

    graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    graph.replay()
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / (iters * inner_loop)


def profile_minimax_allreduce_rms_qk(
    mapping: Mapping,
    op: MiniMaxAllReduceRMS,
    warmup: int = 10,
    iters: int = 100,
    inner_loop: int = 8,
    q_tensor=None,
    k_tensor=None,
    norm_weight_q=None,
    norm_weight_k=None,
    eps: float = 1e-5,
):
    """Profile the fused Q+K minimax allreduce RMS API (forward_qk)."""

    def func():
        for _ in range(inner_loop):
            op.forward_qk(q_tensor, k_tensor, norm_weight_q, norm_weight_k, eps)

    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        func()

    graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    graph.replay()
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / (iters * inner_loop)


def minimax_allreduce_benchmark(
    dtype: str = "bfloat16",
    test_range: str = "256,256000000,10",
    explore_2d: bool = False,
    save_csv: str = None,
    warmup: int = 10,
    iters: int = 100,
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

    inner_loop = 8
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

        mpi_barrier()
        median_us = profile_minimax_allreduce_rms(
            mapping=mapping,
            op=op,
            warmup=warmup,
            iters=iters,
            inner_loop=inner_loop,
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
                            "api": ["single"],
                            "message_size_bytes": [message_size_bytes],
                            "num_tokens": [num_tokens],
                            "hidden_size": [hidden_size],
                            "q_dim": [pd.NA],
                            "k_dim": [pd.NA],
                            "time (us)": [median_us],
                        }
                    ),
                ]
            )
            print(f"num_tokens: {num_tokens}, hidden_size: {hidden_size}, time (us): {median_us}")

    # Q+K fused API benchmark: q_dim=1536, k_dim=128
    num_tokens_qk = sorted({n for n, _ in shape_list})
    for num_tokens in num_tokens_qk:
        q_tensor = torch.ones((num_tokens, QK_Q_DIM), dtype=torch_dtype, device="cuda")
        k_tensor = torch.ones((num_tokens, QK_K_DIM), dtype=torch_dtype, device="cuda")
        norm_weight_q = torch.randn((QK_Q_DIM,), dtype=torch_dtype, device="cuda")
        norm_weight_k = torch.randn((QK_K_DIM,), dtype=torch_dtype, device="cuda")
        message_size_bytes_qk = (
            num_tokens * (QK_Q_DIM + QK_K_DIM) * torch.finfo(torch_dtype).bits // 8
        )
        if message_size_bytes_qk > max_workspace:
            continue

        mpi_barrier()
        median_us_qk = profile_minimax_allreduce_rms_qk(
            mapping=mapping,
            op=op,
            warmup=warmup,
            iters=iters,
            inner_loop=inner_loop,
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            norm_weight_q=norm_weight_q,
            norm_weight_k=norm_weight_k,
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
                            "api": ["qk"],
                            "message_size_bytes": [message_size_bytes_qk],
                            "num_tokens": [num_tokens],
                            "hidden_size": [pd.NA],
                            "q_dim": [QK_Q_DIM],
                            "k_dim": [QK_K_DIM],
                            "time (us)": [median_us_qk],
                        }
                    ),
                ]
            )
            print(
                f"qk: num_tokens: {num_tokens}, q_dim: {QK_Q_DIM}, k_dim: {QK_K_DIM}, "
                f"time (us): {median_us_qk}"
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
    parser.add_argument("--save_csv", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)

    args = parser.parse_args()

    minimax_allreduce_benchmark(
        args.dtype,
        args.range,
        args.explore_2d,
        args.save_csv,
        args.warmup,
        args.iters,
    )
