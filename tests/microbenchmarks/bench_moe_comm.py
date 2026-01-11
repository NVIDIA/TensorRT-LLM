# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# ruff: noqa: E501

"""Unified MoE communication microbenchmark (MPI).

This benchmarks ONLY the communication kernels, specifically:
- Communication.dispatch()
- Communication.combine()

Timing method:
- CUDA events on the default stream
- Region wrapped by cudaProfilerStart/Stop (via torch.cuda.profiler.start/stop)
  so Nsight Compute (ncu) can capture it later.

Launch (examples):

```bash
# Run backend AllGather/ReduceScatter
 python tests/microbenchmarks/bench_moe_comm.py --ep-size 8 --backend ALLGATHER --profile gpt_oss

# Run backend NVLinkOneSided
 python tests/microbenchmarks/bench_moe_comm.cpy --ep-size 8 --backend NVLINK_ONE_SIDED --profile deepseek_v3

# With batch size sweeping
 /scratch/projects/tekit/tests/microbenchmarks/bench_moe_comm.py --ep-size 8 --backend NVLINK_ONE_SIDED -b 1 -e 1024 -f 2

# Run all supported strategies for the given profile
 python tests/microbenchmarks/bench_moe_comm.py --ep-size 8 --profile deepseek_v3

```
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cloudpickle
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm as tllm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe.communication.communication_factory import (
    CommunicationFactory,
)
from tensorrt_llm._utils import local_mpi_rank, mpi_allgather, mpi_barrier, mpi_rank, mpi_world_size
from tensorrt_llm.mapping import Mapping


@dataclass(frozen=True)
class _DummyPretrainedConfig:
    # Only the two attributes used by CommunicationFactory / ModelConfig.torch_dtype.
    hidden_size: int
    torch_dtype: torch.dtype


@dataclass(frozen=True)
class Profile:
    name: str
    hidden_size: int
    top_k: int
    num_experts: int


PROFILES: Dict[str, Profile] = {
    # DeepSeek-V3: hidden_size 7168, router_topk 8 (public config)
    "deepseek_v3": Profile(
        name="deepseek_v3",
        hidden_size=7168,
        top_k=8,
        # Previously: experts_per_rank=32 with recommended ep_size=8 => 256 total experts.
        num_experts=256,
    ),
    # Repo already references "gpt-oss" hidden_size=2880 in the MoE A2A unit test.
    "gpt_oss": Profile(
        name="gpt_oss",
        hidden_size=2880,
        top_k=2,
        num_experts=64,
    ),
}


# TODO: Support tensorrt-llm quantization!
def _torch_dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def _maybe_warn_rank0(msg: str):
    if mpi_rank() == 0:
        print(msg, flush=True)


def _set_device_from_local_rank():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    dev = local_mpi_rank() % torch.cuda.device_count()
    torch.cuda.set_device(dev)
    return dev


def _make_inputs(
    local_num_tokens: int,
    hidden_size: int,
    top_k: int,
    num_experts_total: int,
    act_dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    # Hidden states: payload we want to communicate.
    hidden_states = torch.randn(local_num_tokens, hidden_size, dtype=act_dtype, device=device)
    # We keep scaling factors optional; most strategies can ignore it.
    hidden_states_sf = None
    # Routing IDs: global expert IDs in [0, num_experts_total).
    token_selected_slots = torch.randint(
        0,
        num_experts_total,
        (local_num_tokens, top_k),
        dtype=torch.int32,
        device=device,
    )
    # Router weights/scales.
    # DeepEP expects router weights/topk_weights to be float32.
    token_final_scales = torch.rand(local_num_tokens, top_k, dtype=torch.float32, device=device)
    return hidden_states, hidden_states_sf, token_selected_slots, token_final_scales


def _create_mapping(ep_size: int) -> Mapping:
    rank = mpi_rank()
    return Mapping(
        rank=rank,
        tp_size=ep_size,
        moe_ep_size=ep_size,
        enable_attention_dp=True,
        world_size=ep_size,
    )


def _create_model_config(
    *,
    mapping: Mapping,
    hidden_size: int,
    act_dtype: torch.dtype,
    max_num_tokens_per_rank: int,
) -> ModelConfig:
    # Keep it minimal: just enough fields for CommunicationFactory.
    return ModelConfig(
        pretrained_config=_DummyPretrainedConfig(hidden_size=hidden_size, torch_dtype=act_dtype),
        mapping=mapping,
        max_num_tokens=int(max_num_tokens_per_rank),
        moe_max_num_tokens=int(max_num_tokens_per_rank),
        use_cuda_graph=False,
        use_low_precision_moe_combine=False,
    )


def _profile_start_stop(enabled: bool):
    # Use cudaProfilerStart/Stop (torch calls cudart under the hood).
    if not enabled:
        return
    torch.cuda.profiler.start()


def _profile_stop(enabled: bool):
    if not enabled:
        return
    torch.cuda.profiler.stop()


@contextmanager
def _nvtx_range(msg: str, enabled: bool) -> None:
    """Best-effort NVTX range helper for Nsight Systems timelines."""
    if not enabled:
        yield
        return
    try:
        torch.cuda.nvtx.range_push(msg)
        yield
    finally:
        # Avoid masking the original exception if NVTX isn't available for some reason.
        try:
            torch.cuda.nvtx.range_pop()
        except Exception:
            pass


def _time_dispatch_and_combine(
    strategy: Any,
    *,
    hidden_states: torch.Tensor,
    hidden_states_sf: Optional[torch.Tensor],
    token_selected_slots: torch.Tensor,
    token_final_scales: Optional[torch.Tensor],
    all_rank_num_tokens: List[int],
    warmup: int,
    iters: int,
    profile_wrap: bool,
    nvtx: bool,
    nvtx_prefix: str,
) -> Tuple[float, float]:
    # Returns: (dispatch_us_per_iter, combine_us_per_iter)

    # Some strategies require prepare_dispatch() before dispatch() (e.g., NVLinkTwoSided).
    # We intentionally exclude prepare_dispatch() from timing as requested.
    _ = strategy.prepare_dispatch(token_selected_slots, all_rank_num_tokens, None)

    # Warmup
    with _nvtx_range(f"{nvtx_prefix}:warmup", enabled=nvtx):
        for _ in range(warmup):
            with _nvtx_range("dispatch", enabled=nvtx):
                hs, hs_sf, tss, tfs = strategy.dispatch(
                    hidden_states,
                    hidden_states_sf,
                    token_selected_slots,
                    token_final_scales,
                    all_rank_num_tokens,
                )
            with _nvtx_range("combine", enabled=nvtx):
                _ = strategy.combine(
                    hs,
                    all_rank_max_num_tokens=max(all_rank_num_tokens),
                )

    torch.cuda.synchronize()
    mpi_barrier()

    # Timed loop: use 3 events so we can isolate dispatch vs combine without extra sync.
    start = torch.cuda.Event(enable_timing=True)
    mid = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    dispatch_total_ms = 0.0
    combine_total_ms = 0.0

    _profile_start_stop(profile_wrap)
    with _nvtx_range(f"{nvtx_prefix}:timed", enabled=nvtx):
        for _ in range(iters):
            start.record()
            with _nvtx_range("dispatch", enabled=nvtx):
                hs, hs_sf, tss, tfs = strategy.dispatch(
                    hidden_states,
                    hidden_states_sf,
                    token_selected_slots,
                    token_final_scales,
                    all_rank_num_tokens,
                )
            mid.record()
            with _nvtx_range("combine", enabled=nvtx):
                _ = strategy.combine(
                    hs,
                    all_rank_max_num_tokens=max(all_rank_num_tokens),
                )
            end.record()

            end.synchronize()
            dispatch_total_ms += start.elapsed_time(mid)
            combine_total_ms += mid.elapsed_time(end)
    _profile_stop(profile_wrap)

    dispatch_ms = dispatch_total_ms / iters
    combine_ms = combine_total_ms / iters
    return dispatch_ms * 1e3, combine_ms * 1e3


def _gather_stats_us(x_us: float) -> Dict[str, float]:
    xs = mpi_allgather(float(x_us))
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    return {
        "min": xs_sorted[0],
        "p50": xs_sorted[n // 2],
        "max": xs_sorted[-1],
        "avg": sum(xs_sorted) / n,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified MoE communication microbenchmark (MPI).")
    parser.add_argument(
        "--ep-size",
        type=int,
        default=None,
        help="Number of MPI worker ranks to spawn via MPIPoolExecutor.",
    )
    parser.add_argument(
        "--backend",
        type=lambda s: str(s).upper(),
        default=None,
        choices=[
            "ALLGATHER",
            "NVLINK_ONE_SIDED",
            "NVLINK_TWO_SIDED",
            "DEEPEPLOWLATENCY",
        ],
        help="Which communication backend to benchmark (default: run all backends).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="deepseek_v3",
        choices=sorted(PROFILES.keys()),
        help="Optional named profile to provide defaults for hidden_size/top_k/num_experts.",
    )
    parser.add_argument("--hidden-size", type=int, default=None, help="Custom hidden size.")
    # Sizes to scan (NCCL-tests style, adapted from bytes -> local_batch_size in tokens).
    parser.add_argument(
        "-b",
        "--minbatch",
        type=int,
        default=640,
        help="Minimum local_batch_size (tokens) to start with. Default: 640. For single size, pass -b N (or -e N).",
    )
    parser.add_argument(
        "-e",
        "--maxbatch",
        type=int,
        default=None,
        help="Maximum local_batch_size (tokens) to end at. For single size, pass -e N (or -b N).",
    )
    # Increments can be either fixed or a multiplication factor. Only one should be used.
    parser.add_argument(
        "-i",
        "--stepbatch",
        type=int,
        default=None,
        help="Fixed increment between local_batch_size values (tokens). Default: 128.",
    )
    parser.add_argument(
        "-f",
        "--stepfactor",
        type=float,
        default=None,
        help="Multiplication factor between local_batch_size values. Default: disabled.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Custom router top-k.")
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Total number of experts.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Activation dtype for the benchmark payload.",
    )
    parser.add_argument("--iters", type=int, default=200, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument(
        "--max-num-tokens-per-rank",
        type=int,
        default=None,
        help="Max tokens per rank for workspace allocation (defaults to max scanned local_batch_size).",
    )
    parser.add_argument(
        "--no-profiler-wrap",
        action="store_true",
        help="Disable cudaProfilerStart/Stop wrapping (useful when profiler is unavailable).",
    )
    parser.add_argument(
        "--no-nvtx",
        action="store_true",
        help="Disable NVTX ranges (useful to minimize profiler overhead).",
    )
    return parser.parse_args()


def _iter_local_batch_sizes(args: argparse.Namespace) -> List[int]:
    scanning_enabled = any(
        x is not None for x in (args.minbatch, args.maxbatch, args.stepbatch, args.stepfactor)
    )
    if not scanning_enabled:
        raise ValueError("Must specify -b/--minbatch and/or -e/--maxbatch (tokens).")

    if args.stepbatch is not None and args.stepfactor is not None:
        raise ValueError("Only one of -i/--stepbatch or -f/--stepfactor should be used.")

    if args.minbatch is None and args.maxbatch is None:
        raise ValueError("Must specify at least one of -b/--minbatch or -e/--maxbatch.")

    # For single-size mode, allowing -b N or -e N (or both).
    minb = int(args.minbatch) if args.minbatch is not None else int(args.maxbatch)
    if minb <= 0:
        raise ValueError("--minbatch must be > 0")

    maxb = int(args.maxbatch) if args.maxbatch is not None else minb
    if maxb < minb:
        raise ValueError("--maxbatch must be >= --minbatch")

    if args.stepfactor is not None:
        factor = float(args.stepfactor)
        if factor <= 1.0:
            raise ValueError("--stepfactor must be > 1.0")
        out: List[int] = []
        cur = float(minb)
        while True:
            v = int(cur)
            if v > maxb:
                break
            if v > 0 and (not out or out[-1] != v):
                out.append(v)
            cur *= factor
        return out

    step = int(args.stepbatch or 128)
    if step <= 0:
        raise ValueError("--stepbatch must be > 0")
    return list(range(minb, maxb + 1, step))


def _resolve_profile_args(args: argparse.Namespace) -> Tuple[int, int, int, int]:
    """Returns (hidden_size, local_num_tokens, top_k, num_experts_total)."""
    local_num_tokens = _iter_local_batch_sizes(args)[0]

    # If a profile is provided, it supplies defaults; any explicit CLI values override.
    if args.profile is not None:
        prof = PROFILES[args.profile]
        hidden_size = int(args.hidden_size or prof.hidden_size)
        top_k = int(args.top_k or prof.top_k)
        num_experts_total = int(args.num_experts or prof.num_experts)
        return hidden_size, local_num_tokens, top_k, num_experts_total

    # No profile: all fields must be provided explicitly.
    if args.hidden_size is None or args.top_k is None or args.num_experts is None:
        raise ValueError(
            "No --profile specified; must provide --hidden-size, --top-k, --num-experts."
        )
    hidden_size = int(args.hidden_size)
    top_k = int(args.top_k)
    num_experts_total = int(args.num_experts)
    return hidden_size, local_num_tokens, top_k, num_experts_total


def _run_benchmark_worker_under_current_mpi(args: argparse.Namespace) -> None:
    # Keep benchmark output clean.
    tllm.logger.set_level("error")
    # MPI-spawned workers may not inherit the parent's mutated environment reliably.
    # Opt-in to DeepEP backends by default (does not override an explicit user setting).
    os.environ.setdefault("TRTLLM_CAN_USE_DEEP_EP", "1")

    ep_size = mpi_world_size()
    rank = mpi_rank()
    _ = _set_device_from_local_rank()
    device = torch.device("cuda")

    hidden_size, _, top_k, num_experts_total = _resolve_profile_args(args)
    local_batch_sizes = _iter_local_batch_sizes(args)
    act_dtype = _torch_dtype(args.dtype)
    nvtx = not bool(args.no_nvtx)

    if (args.backend is None or args.backend == "DEEPEPLOWLATENCY") and act_dtype != torch.bfloat16:
        _maybe_warn_rank0("[bench_moe_comm] DeepEP low latency requires bf16; will be skipped.")

    max_scanned_tokens = max(local_batch_sizes)
    max_num_tokens_per_rank = int(args.max_num_tokens_per_rank or max_scanned_tokens)
    if max_num_tokens_per_rank < max_scanned_tokens:
        raise ValueError("--max-num-tokens-per-rank must be >= --local-batch-size")

    mapping = _create_mapping(ep_size)

    if num_experts_total % ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts_total}) must be divisible by ep_size ({ep_size})"
        )
    experts_per_rank = num_experts_total // ep_size
    num_slots = num_experts_total

    if rank == 0:
        print(
            json.dumps(
                {
                    "bench": "bench_moe_comm",
                    "launcher": "spawn",
                    "profile": args.profile,
                    "backend": args.backend,
                    "ep_size": ep_size,
                    "hidden_size": hidden_size,
                    "local_batch_size": local_batch_sizes,
                    "top_k": top_k,
                    "dtype": str(act_dtype),
                    "experts_per_rank": experts_per_rank,
                    "num_experts_total": num_experts_total,
                    "max_num_tokens_per_rank": max_num_tokens_per_rank,
                    "device_count": torch.cuda.device_count(),
                },
                indent=2,
            ),
            flush=True,
        )

    backends = (
        ["ALLGATHER", "NVLINK_ONE_SIDED", "NVLINK_TWO_SIDED", "DEEPEPLOWLATENCY"]
        if args.backend is None
        else [args.backend]
    )

    for backend in backends:
        if backend == "DEEPEPLOWLATENCY" and act_dtype != torch.bfloat16:
            continue

        try:
            model_config = _create_model_config(
                mapping=mapping,
                hidden_size=hidden_size,
                act_dtype=act_dtype,
                max_num_tokens_per_rank=max_num_tokens_per_rank,
            )

            # Avoid env-var indirection: directly construct the requested backend.
            strategy = CommunicationFactory._create_forced_method(  # pylint: disable=protected-access
                backend,
                model_config,
                num_experts_total,
                num_slots,
                top_k,
                experts_per_rank,
                payload_in_workspace=False,
                alltoall_result_do_sum=True,
            )

            if strategy is None:
                _maybe_warn_rank0(f"[bench_moe_comm] Skipping {backend}: factory returned None.")
                mpi_barrier()
                continue
        except Exception as e:
            _maybe_warn_rank0(f"[bench_moe_comm] Skipping {backend}: {type(e).__name__}: {e}")
            mpi_barrier()
            continue

        for local_num_tokens in local_batch_sizes:
            all_rank_num_tokens = mpi_allgather(int(local_num_tokens))
            try:
                if not strategy.is_workload_feasible(all_rank_num_tokens, num_chunks=1):
                    _maybe_warn_rank0(
                        f"[bench_moe_comm] Skipping {backend} @ local_batch_size={local_num_tokens}: workload not feasible."
                    )
                    mpi_barrier()
                    continue
            except Exception as e:
                _maybe_warn_rank0(
                    f"[bench_moe_comm] Skipping {backend} @ local_batch_size={local_num_tokens}: feasibility check failed: {e}"
                )
                mpi_barrier()
                continue

            hidden_states, hidden_states_sf, token_selected_slots, token_final_scales = (
                _make_inputs(
                    local_num_tokens,
                    hidden_size,
                    top_k,
                    num_experts_total,
                    act_dtype,
                    device,
                )
            )

            mpi_barrier()
            torch.cuda.synchronize()

            with _nvtx_range(f"{backend}:local_batch_size={int(local_num_tokens)}", enabled=nvtx):
                nvtx_prefix = f"{backend}:local_batch_size={int(local_num_tokens)}"
                dispatch_us, combine_us = _time_dispatch_and_combine(
                    strategy,
                    hidden_states=hidden_states,
                    hidden_states_sf=hidden_states_sf,
                    token_selected_slots=token_selected_slots,
                    token_final_scales=token_final_scales,
                    all_rank_num_tokens=all_rank_num_tokens,
                    warmup=int(args.warmup),
                    iters=int(args.iters),
                    profile_wrap=(not args.no_profiler_wrap),
                    nvtx=nvtx,
                    nvtx_prefix=nvtx_prefix,
                )

            dispatch_stats = _gather_stats_us(dispatch_us)
            combine_stats = _gather_stats_us(combine_us)

            if rank == 0:
                print(
                    json.dumps(
                        {
                            "backend": backend,
                            "local_batch_size": int(local_num_tokens),
                            "dispatch_us": dispatch_stats,
                            "combine_us": combine_stats,
                        },
                        indent=2,
                    ),
                    flush=True,
                )

            mpi_barrier()

    return


def _spawn_worker_main(args_blob: bytes) -> List[Dict[str, Any]]:
    """Worker entrypoint for MPIPoolExecutor.

    Note: MPIPoolExecutor typically serializes callables by name; for scripts executed
    as __main__, we set MPI's pickler to cloudpickle in the parent so this function is
    available to workers (same pattern as our unit tests).
    """
    args = pickle.loads(args_blob)
    # In spawned workers, we are already inside an MPI world of size == ep_size.
    _run_benchmark_worker_under_current_mpi(args)
    return []


def main() -> None:
    args = parse_args()

    # Important: parent is NOT part of the worker MPI world; do not call mpi_barrier here.
    # Make functions/classes in this script pickleable to spawned workers.
    # (Same pattern used in our MPI unit tests, but adapted for a script entrypoint.)
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    MPI.pickle.__init__(  # type: ignore[attr-defined]
        cloudpickle.dumps,
        cloudpickle.loads,
        pickle.HIGHEST_PROTOCOL,
    )

    ep_size = int(args.ep_size or 8)
    if ep_size <= 0:
        raise ValueError("--ep-size must be > 0")

    if mpi_world_size() != 1:
        raise RuntimeError("bench_moe_comm should be run from a non-MPI parent (world_size==1).")

    if mpi_rank() == 0:
        print(
            json.dumps(
                {
                    "bench": "bench_moe_comm",
                    "launcher": "spawn",
                    "ep_size": ep_size,
                    "profile": args.profile,
                    "backend": args.backend,
                },
                indent=2,
            ),
            flush=True,
        )

    args_blob = cloudpickle.dumps(args)
    with MPIPoolExecutor(max_workers=ep_size) as executor:
        # Map the same args to all workers; each worker uses its own mpi_rank() and participates
        # in collectives within its spawned MPI world.
        _ = list(executor.map(_spawn_worker_main, [args_blob] * ep_size))


if __name__ == "__main__":
    # Make sure ranks don't accidentally serialize execution.
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    # Opt-in to DeepEP backends by default. This does not override an explicit user setting.
    os.environ.setdefault("TRTLLM_CAN_USE_DEEP_EP", "1")
    main()
