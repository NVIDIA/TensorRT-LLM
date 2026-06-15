# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
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
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.custom_ops.userbuffers_custom_ops import \
    copy_to_userbuffers
from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             Distributed,
                                             userbuffers_allreduce_finalize)
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
    allreduce_instance=None,
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

    allreduce = allreduce_instance or AllReduce(mapping=mapping,
                                                strategy=strategy)
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
            elif strategy in (AllReduceStrategy.NCCL_SYMMETRIC,
                              AllReduceStrategy.NCCL, AllReduceStrategy.AUTO):
                try:
                    from tensorrt_llm.bindings.internal.thop import BufferKind
                    window_out, actual_kind = torch.ops.trtllm.allocate_output(
                        input, int(BufferKind.NCCL_WINDOW), mapping.tp_group)
                    if actual_kind == int(BufferKind.NCCL_WINDOW):
                        window_out.copy_(input)
                        input_for_profile = window_out
                except RuntimeError:
                    pass

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


# ── nccl-tests style comprehensive benchmark (--benchmark mode) ──────────────

_STRATEGY_MAP = {
    "NCCL": AllReduceStrategy.NCCL,
    "NCCL_SYMMETRIC": AllReduceStrategy.NCCL_SYMMETRIC,
    "UB": AllReduceStrategy.UB,
    "ONESHOT": AllReduceStrategy.ONESHOT,
    "TWOSHOT": AllReduceStrategy.TWOSHOT,
    "AUTO": AllReduceStrategy.AUTO,
    "MNNVL": AllReduceStrategy.MNNVL,
}
_UB_STRATEGIES = {AllReduceStrategy.NCCL_SYMMETRIC, AllReduceStrategy.UB}
_FUSION_MAP = {
    "NONE":
    AllReduceFusionOp.NONE,
    "RESIDUAL_RMS_NORM":
    AllReduceFusionOp.RESIDUAL_RMS_NORM,
    "RESIDUAL_RMS_NORM_QUANT_FP8":
    AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
    "RESIDUAL_RMS_NORM_QUANT_NVFP4":
    AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
}


def _fmt_size(nbytes):
    """Format byte count as human-readable string (e.g. 256B, 4K, 1M, 2G)."""
    if nbytes < 1024:
        return f"{nbytes}B"
    elif nbytes < 1024**2:
        v = nbytes / 1024
        return f"{v:.0f}K" if nbytes % 1024 == 0 else f"{v:.1f}K"
    elif nbytes < 1024**3:
        v = nbytes / 1024**2
        return f"{v:.0f}M" if nbytes % (1024**2) == 0 else f"{v:.2f}M"
    else:
        v = nbytes / 1024**3
        return f"{v:.0f}G" if nbytes % (1024**3) == 0 else f"{v:.2f}G"


def _profile_ub(mapping,
                dist,
                allreduce,
                fusion,
                input,
                residual,
                norm,
                scale,
                enable_cudagraph=False,
                inner_loop=200,
                outer_loop=10):
    """Profile UB allreduce kernel only (copy_to_ub and finalize are one-shot)."""
    allreduce_params = AllReduceParams(fusion_op=fusion,
                                       residual=residual,
                                       norm_weight=norm.weight,
                                       eps=norm.variance_epsilon,
                                       scale=scale,
                                       bias=None)

    # Copy input into user-buffer memory once (simulates matmul_to_ub in real flow)
    ub_input = copy_to_userbuffers(input)

    def func(loop_num=inner_loop):
        for _ in range(loop_num):
            output = allreduce(ub_input, all_reduce_params=allreduce_params)
        return output

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(outer_loop)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(outer_loop)]
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # warmup
        for _ in range(4):
            func(loop_num=1)
        if enable_cudagraph:
            with torch.cuda.graph(graph, stream=stream):
                func()
        dist.barrier()
        delay_kernel(20000, stream)
        torch.cuda.synchronize()
        for i in range(outer_loop):
            starts[i].record(stream)
            if enable_cudagraph:
                graph.replay()
            else:
                func()
            stops[i].record(stream)
    torch.cuda.synchronize()
    # Finalize once to sync (simulates userbuffers_allreduce_finalize in real flow)
    output = func(loop_num=1)
    userbuffers_allreduce_finalize(output[-1])
    runtimes = [starts[i].elapsed_time(stops[i]) for i in range(outer_loop)]
    return sorted(runtimes)[len(runtimes) // 2] / inner_loop * 1000.0


def _print_table(fusion_name, strategy_names, rows, world_size):
    W_S, W_T, W_H, W_V, W_B = 10, 6, 6, 10, 16
    n = len(strategy_names)
    print(flush=True)
    print(
        f"# Fusion: {fusion_name}    world_size={world_size}    "
        f"algbw = size / time (GB/s)",
        flush=True)
    print("#", flush=True)
    fixed = f"{'size':>{W_S}}  {'ntok':>{W_T}}  {'hdim':>{W_H}}"
    sh = "  ".join(f"{s:^{W_V * 2 + 2}}" for s in strategy_names)
    print(f"# {fixed}  {sh}  {'BEST':>{W_B}}", flush=True)
    pad = " " * (W_S + 2 + W_T + 2 + W_H)
    mh = "  ".join(f"{'time(us)':>{W_V}}  {'algbw':>{W_V}}"
                   for _ in strategy_names)
    print(f"# {pad}  {mh}  {' ':>{W_B}}", flush=True)
    tw = 2 + W_S + 2 + W_T + 2 + W_H + 2 + n * (W_V * 2 + 2) + (n -
                                                                1) * 2 + 2 + W_B
    print("#" + "-" * (tw - 1), flush=True)
    for row in rows:
        prefix = (f"  {row['size_human']:>{W_S}}  "
                  f"{row['num_tokens']:>{W_T}}  "
                  f"{row['hidden_size']:>{W_H}}")
        vals, best_name, best_time = [], "N/A", float("inf")
        for s in strategy_names:
            t, bw = row.get(f"{s}_time"), row.get(f"{s}_algbw")
            if t is not None:
                vals.append(f"{t:>{W_V}.2f}  {bw:>{W_V}.2f}")
                if t < best_time:
                    best_time, best_name = t, s
            else:
                vals.append(f"{'N/A':>{W_V}}  {'N/A':>{W_V}}")
        print(f"{prefix}  {'  '.join(vals)}  {best_name:>{W_B}}", flush=True)


def allreduce_benchmark_all(
    dtype='bfloat16',
    test_range="256,268435456,2",
    explore_2d=False,
    enable_cudagraph=False,
    strategy_names=None,
    fusion_names=None,
    inner_loop=200,
    outer_loop=10,
    save_csv=None,
):
    """Comprehensive benchmark: one table per fusion, all strategies side by side."""
    import csv as csv_mod

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
        raise RuntimeError("Benchmark requires mpi_world_size > 1")

    torch_dtype = tllm._utils.str_dtype_to_torch(dtype)
    elem_size = torch.finfo(torch_dtype).bits // 8

    # Enable MNNVL testing on single-node (bypasses multi-node NVLink check)
    os.environ["TLLM_TEST_MNNVL"] = "1"

    # strategies
    if strategy_names is None:
        strategy_names = [
            "NCCL", "NCCL_SYMMETRIC", "UB", "ONESHOT", "TWOSHOT", "AUTO",
            "MNNVL"
        ]
    strategies = [_STRATEGY_MAP[s] for s in strategy_names]

    # fusions
    if fusion_names is None:
        fusion_names = list(_FUSION_MAP.keys())
    fusions = []
    for f in fusion_names:
        fop = _FUSION_MAP[f]
        if fop == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4 and sm_version < 100:
            if rank == 0:
                print(f"[WARN] {f} requires SM100+, skipping.", flush=True)
            continue
        fusions.append((f, fop))

    # shapes
    if explore_2d:
        num_tokens_list = [
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
        ]
        hidden_size_list = [128, 256, 512, 1024, 2048, 4096, 8192]
        shape_list = list(product(num_tokens_list, hidden_size_list))
    else:
        min_bytes, max_bytes, ratio = [int(i) for i in test_range.split(",")]
        shape_list = []
        nbytes = min_bytes
        while nbytes <= max_bytes:
            total_elems = nbytes // elem_size
            if total_elems <= 4096:
                shape_list.append((1, max(total_elems, 1)))
            else:
                shape_list.append((total_elems // 4096, 4096))
            nbytes *= ratio

    # init user-buffers
    need_ub = bool(_UB_STRATEGIES & set(strategies))
    if need_ub:
        if ub.ub_supported():
            max_elems = max(s[0] * s[1] for s in shape_list)
            ub.initialize_userbuffers_manager(world_size, 1, 1, rank,
                                              torch.cuda.device_count(),
                                              max_elems * elem_size)
        else:
            if rank == 0:
                print("[WARN] ub not supported, skipping UB-based strategies.",
                      flush=True)
            strategies = [s for s in strategies if s not in _UB_STRATEGIES]
            strategy_names = [s.name for s in strategies]

    # create AllReduce instances
    ar_instances = {}
    for strat in strategies:
        try:
            ar_instances[strat] = AllReduce(mapping=mapping,
                                            strategy=strat,
                                            dtype=torch_dtype)
        except Exception as e:
            if rank == 0:
                print(f"[WARN] Cannot init {strat.name}: {e}", flush=True)
    strategies = [s for s in strategies if s in ar_instances]
    strategy_names = [s.name for s in strategies]

    max_workspace = CustomAllReduceHelper.max_workspace_size_auto(
        mapping.tp_size)

    if rank == 0:
        print(f"\n{'=' * 80}", flush=True)
        print("  TRT-LLM AllReduce Benchmark", flush=True)
        print(
            f"  world_size={world_size}  dtype={dtype}  SM={sm_version}"
            f"  cudagraph={enable_cudagraph}"
            f"  inner={inner_loop}  outer={outer_loop}",
            flush=True)
        print(f"  Strategies : {', '.join(strategy_names)}", flush=True)
        print(f"  Fusions    : {', '.join(f for f, _ in fusions)}", flush=True)
        print(f"{'=' * 80}", flush=True)

    csv_rows = []

    for fusion_name, fusion_op in fusions:
        table_rows = []
        for num_tokens, hidden_size in shape_list:
            msg_bytes = num_tokens * hidden_size * elem_size
            inp = torch.ones((num_tokens, hidden_size),
                             dtype=torch_dtype,
                             device="cuda")
            res = torch.randn_like(inp)
            norm = RMSNorm(hidden_size=hidden_size, dtype=torch_dtype,
                           eps=1e-5).cuda()
            norm.weight.data.copy_(
                torch.randn((hidden_size, ), dtype=torch_dtype, device="cuda"))
            scale = torch.tensor(1.0, dtype=torch.float32).cuda()

            row = dict(size_human=_fmt_size(msg_bytes),
                       num_tokens=num_tokens,
                       hidden_size=hidden_size,
                       size_bytes=msg_bytes)

            for strat in strategies:
                sn = strat.name
                # skip invalid combos
                skip = False
                if strat == AllReduceStrategy.TWOSHOT and num_tokens < world_size:
                    skip = True
                elif strat in (AllReduceStrategy.ONESHOT, AllReduceStrategy.TWOSHOT) \
                        and msg_bytes > max_workspace:
                    skip = True
                elif strat == AllReduceStrategy.UB and fusion_op == AllReduceFusionOp.NONE:
                    skip = True

                if skip:
                    row[f"{sn}_time"] = row[f"{sn}_algbw"] = None
                else:
                    try:
                        if strat == AllReduceStrategy.UB:
                            t_us = _profile_ub(mapping, dist,
                                               ar_instances[strat], fusion_op,
                                               inp, res, norm, scale,
                                               enable_cudagraph, inner_loop,
                                               outer_loop)
                        else:
                            t_us = profile_allreduce(
                                mapping=mapping,
                                dist=dist,
                                enable_cudagraph=enable_cudagraph,
                                inner_loop=inner_loop,
                                outer_loop=outer_loop,
                                fusion=fusion_op,
                                input=inp,
                                residual=res,
                                norm=norm,
                                scale=scale,
                                allreduce_instance=ar_instances[strat]) * 1000.0
                        row[f"{sn}_time"] = t_us
                        row[f"{sn}_algbw"] = msg_bytes / (t_us / 1e6) / 1e9
                    except Exception as e:
                        if rank == 0:
                            print(
                                f"  [SKIP] {sn} @ {_fmt_size(msg_bytes)}: {e}",
                                flush=True)
                        row[f"{sn}_time"] = row[f"{sn}_algbw"] = None

                csv_rows.append({
                    "world_size": world_size,
                    "dtype": dtype,
                    "fusion": fusion_name,
                    "num_tokens": num_tokens,
                    "hidden_size": hidden_size,
                    "size_bytes": msg_bytes,
                    "strategy": sn,
                    "time_us": row[f"{sn}_time"] or 0.0,
                    "algbw_GBps": row[f"{sn}_algbw"] or 0.0,
                })
            table_rows.append(row)

        if rank == 0:
            _print_table(fusion_name, strategy_names, table_rows, world_size)

    # summary
    if rank == 0:
        print(f"\n{'=' * 80}", flush=True)
        print("  Summary: peak algbw (GB/s) per strategy per fusion",
              flush=True)
        print(f"{'=' * 80}", flush=True)
        hdr = f"  {'fusion':<35s}" + "".join(f"  {s:>14s}"
                                             for s in strategy_names)
        print(hdr, flush=True)
        print("  " + "-" * (len(hdr) - 2), flush=True)
        for fn, _ in fusions:
            line = f"  {fn:<35s}"
            for sn in strategy_names:
                bws = [
                    r["algbw_GBps"] for r in csv_rows if r["fusion"] == fn
                    and r["strategy"] == sn and r["algbw_GBps"] > 0
                ]
                line += f"  {max(bws) if bws else 0.0:>14.2f}"
            print(line, flush=True)
        print(flush=True)

    if rank == 0 and save_csv and csv_rows:
        with open(save_csv, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Results saved to {save_csv}", flush=True)


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
    parser.add_argument("--benchmark",
                        action="store_true",
                        default=False,
                        help="Run comprehensive benchmark across all backends "
                        "with nccl-tests style output")
    parser.add_argument("--profile_gemm_allreduce",
                        action="store_true",
                        default=False)

    args = parser.parse_args()

    if args.benchmark:
        allreduce_benchmark_all(
            dtype=args.dtype,
            test_range=args.range,
            explore_2d=args.explore_2d,
            enable_cudagraph=args.enable_cudagraph,
            save_csv=args.save_csv,
        )
    else:
        allreduce_benchmark(
            args.dtype,
            args.range,
            args.enable_cudagraph,
            args.explore_2d,
            args.save_csv,
            args.enable_auto,
            args.profile_gemm_allreduce,
        )
