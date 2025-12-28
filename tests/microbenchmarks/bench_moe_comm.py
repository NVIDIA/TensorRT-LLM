# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Spawn 8 MPI ranks without using mpirun directly (same mechanism as our unit tests)
python tests/microbenchmarks/bench_moe_comm.py --ep-size 8 --strategy allgather --profile gpt_oss

# NVLink one-sided (requires MNNVL/NVLink support)
python tests/microbenchmarks/bench_moe_comm.py --ep-size 8 --strategy nvlink_one_sided --profile deepseek_v3

# Run all supported strategies for the given profile
python tests/microbenchmarks/bench_moe_comm.py --ep-size 8 --strategy all --profile deepseek_v3
```
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
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
    local_batch_size: int
    top_k: int
    # Note: EP size comes from MPI world size; this is just documentation / sanity-check.
    recommended_ep_size: Optional[int] = None
    # Routing model params (benchmark needs a total expert id space).
    experts_per_rank: int = 32


PROFILES: Dict[str, Profile] = {
    # DeepSeek-V3: hidden_size 7168, router_topk 8 (public config)
    "deepseek_v3": Profile(
        name="deepseek_v3",
        hidden_size=7168,
        local_batch_size=1024,
        top_k=8,
        recommended_ep_size=8,
        experts_per_rank=32,
    ),
    # Repo already references "gpt-oss" hidden_size=2880 in the MoE A2A unit test.
    "gpt_oss": Profile(
        name="gpt_oss",
        hidden_size=2880,
        local_batch_size=640,
        top_k=2,
        recommended_ep_size=8,
        experts_per_rank=8,
    ),
}


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
    token_final_scales = torch.rand(local_num_tokens, top_k, dtype=act_dtype, device=device)
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
) -> Tuple[float, float]:
    # Returns: (dispatch_ms_per_iter, combine_ms_per_iter)

    # Some strategies require prepare_dispatch() before dispatch() (e.g., NVLinkTwoSided).
    # We intentionally exclude prepare_dispatch() from timing as requested.
    _ = strategy.prepare_dispatch(token_selected_slots, all_rank_num_tokens, None)

    # Warmup
    for _ in range(warmup):
        hs, hs_sf, tss, tfs = strategy.dispatch(
            hidden_states,
            hidden_states_sf,
            token_selected_slots,
            token_final_scales,
            all_rank_num_tokens,
        )
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
    for _ in range(iters):
        start.record()
        hs, hs_sf, tss, tfs = strategy.dispatch(
            hidden_states,
            hidden_states_sf,
            token_selected_slots,
            token_final_scales,
            all_rank_num_tokens,
        )
        mid.record()
        _ = strategy.combine(
            hs,
            all_rank_max_num_tokens=max(all_rank_num_tokens),
        )
        end.record()

        end.synchronize()
        dispatch_total_ms += start.elapsed_time(mid)
        combine_total_ms += mid.elapsed_time(end)
    _profile_stop(profile_wrap)

    return dispatch_total_ms / iters, combine_total_ms / iters


def _gather_stats_ms(x_ms: float) -> Dict[str, float]:
    xs = mpi_allgather(float(x_ms))
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
        "--strategy",
        type=str,
        default="all",
        choices=[
            "all",
            "allgather",
            "nvlink_one_sided",
            "nvlink_two_sided",
            "deep_ep_low_latency",
        ],
        help="Which communication strategy to benchmark.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="gpt_oss",
        choices=sorted(PROFILES.keys()) + ["custom"],
        help="Named profile (hidden_size + local_batch_size + top_k).",
    )
    parser.add_argument("--hidden-size", type=int, default=None, help="Custom hidden size.")
    parser.add_argument(
        "--local-batch-size", type=int, default=None, help="Custom local token count per rank."
    )
    parser.add_argument("--top-k", type=int, default=None, help="Custom router top-k.")
    parser.add_argument(
        "--experts-per-rank",
        type=int,
        default=None,
        help="Experts per rank (total experts = ep_size * experts_per_rank).",
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
        help="Max tokens per rank for workspace allocation (defaults to local_batch_size).",
    )
    parser.add_argument(
        "--nvlink-two-sided-do-sum",
        action="store_true",
        help="Enable reduction inside NVLinkTwoSided combine (typical path).",
    )
    parser.add_argument(
        "--no-profiler-wrap",
        action="store_true",
        help="Disable cudaProfilerStart/Stop wrapping (useful when profiler is unavailable).",
    )
    return parser.parse_args()


def _resolve_profile_args(args: argparse.Namespace, ep_size: int) -> Tuple[int, int, int, int]:
    if args.profile != "custom":
        prof = PROFILES[args.profile]
        hidden_size = prof.hidden_size
        local_batch_size = prof.local_batch_size
        top_k = prof.top_k
        experts_per_rank = prof.experts_per_rank
        if prof.recommended_ep_size is not None and prof.recommended_ep_size != ep_size:
            _maybe_warn_rank0(
                f"[bench_moe_comm] WARNING: profile {prof.name} recommends ep_size={prof.recommended_ep_size}, "
                f"but current ep_size is ep_size={ep_size}. Continuing."
            )
        return hidden_size, local_batch_size, top_k, experts_per_rank

    if args.hidden_size is None or args.local_batch_size is None or args.top_k is None:
        raise ValueError("--profile custom requires --hidden-size, --local-batch-size, --top-k")
    hidden_size = int(args.hidden_size)
    local_batch_size = int(args.local_batch_size)
    top_k = int(args.top_k)
    experts_per_rank = int(args.experts_per_rank or 32)
    return hidden_size, local_batch_size, top_k, experts_per_rank


def _run_benchmark_worker_under_current_mpi(args: argparse.Namespace) -> None:
    # Keep benchmark output clean.
    tllm.logger.set_level("error")

    ep_size = mpi_world_size()
    rank = mpi_rank()
    _ = _set_device_from_local_rank()
    device = torch.device("cuda")

    hidden_size, local_batch_size, top_k, experts_per_rank = _resolve_profile_args(args, ep_size)
    experts_per_rank = int(args.experts_per_rank or experts_per_rank)
    act_dtype = _torch_dtype(args.dtype)

    if args.strategy in ("deep_ep_low_latency", "all") and act_dtype != torch.bfloat16:
        _maybe_warn_rank0("[bench_moe_comm] DeepEP low latency requires bf16; will be skipped.")

    max_num_tokens_per_rank = int(args.max_num_tokens_per_rank or local_batch_size)
    if max_num_tokens_per_rank < local_batch_size:
        raise ValueError("--max-num-tokens-per-rank must be >= --local-batch-size")

    all_rank_num_tokens = mpi_allgather(int(local_batch_size))
    mapping = _create_mapping(ep_size)

    num_experts_total = ep_size * experts_per_rank
    num_slots = num_experts_total

    hidden_states, hidden_states_sf, token_selected_slots, token_final_scales = _make_inputs(
        local_batch_size,
        hidden_size,
        top_k,
        num_experts_total,
        act_dtype,
        device,
    )

    if rank == 0:
        print(
            json.dumps(
                {
                    "bench": "bench_moe_comm",
                    "launcher": "spawn",
                    "profile": args.profile,
                    "strategy": args.strategy,
                    "ep_size": ep_size,
                    "hidden_size": hidden_size,
                    "local_batch_size": local_batch_size,
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

    strategies = (
        ["allgather", "nvlink_one_sided", "nvlink_two_sided", "deep_ep_low_latency"]
        if args.strategy == "all"
        else [args.strategy]
    )

    mpi_barrier()
    torch.cuda.synchronize()

    for strat in strategies:
        if strat == "deep_ep_low_latency" and act_dtype != torch.bfloat16:
            continue

        try:
            model_config = _create_model_config(
                mapping=mapping,
                hidden_size=hidden_size,
                act_dtype=act_dtype,
                max_num_tokens_per_rank=max_num_tokens_per_rank,
            )

            # Select the desired comm strategy via the same mechanism used in production code.
            forced = "DEEPEPLOWLATENCY" if strat == "deep_ep_low_latency" else strat.upper()
            old_forced = os.environ.get("TRTLLM_FORCE_COMM_METHOD")
            os.environ["TRTLLM_FORCE_COMM_METHOD"] = forced
            try:
                strategy = CommunicationFactory.create_strategy(
                    model_config,
                    num_experts=num_experts_total,
                    num_slots=num_slots,
                    top_k=top_k,
                    expert_size_per_partition=experts_per_rank,
                    payload_in_workspace=False,
                    alltoall_result_do_sum=bool(args.nvlink_two_sided_do_sum),
                )
            finally:
                if old_forced is None:
                    os.environ.pop("TRTLLM_FORCE_COMM_METHOD", None)
                else:
                    os.environ["TRTLLM_FORCE_COMM_METHOD"] = old_forced

            if strategy is None:
                _maybe_warn_rank0(f"[bench_moe_comm] Skipping {strat}: factory returned None.")
                mpi_barrier()
                continue
        except Exception as e:
            _maybe_warn_rank0(f"[bench_moe_comm] Skipping {strat}: {type(e).__name__}: {e}")
            mpi_barrier()
            continue

        try:
            if not strategy.is_workload_feasible(all_rank_num_tokens, num_chunks=1):
                _maybe_warn_rank0(f"[bench_moe_comm] Skipping {strat}: workload not feasible.")
                mpi_barrier()
                continue
        except Exception as e:
            _maybe_warn_rank0(f"[bench_moe_comm] Skipping {strat}: feasibility check failed: {e}")
            mpi_barrier()
            continue

        mpi_barrier()
        torch.cuda.synchronize()

        dispatch_ms, combine_ms = _time_dispatch_and_combine(
            strategy,
            hidden_states=hidden_states,
            hidden_states_sf=hidden_states_sf,
            token_selected_slots=token_selected_slots,
            token_final_scales=token_final_scales,
            all_rank_num_tokens=all_rank_num_tokens,
            warmup=int(args.warmup),
            iters=int(args.iters),
            profile_wrap=(not args.no_profiler_wrap),
        )

        dispatch_stats = _gather_stats_ms(dispatch_ms)
        combine_stats = _gather_stats_ms(combine_ms)

        if rank == 0:
            print(
                json.dumps(
                    {
                        "strategy": strat,
                        "dispatch_ms": dispatch_stats,
                        "combine_ms": combine_stats,
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

    default_ep_size = (
        PROFILES.get(args.profile).recommended_ep_size if args.profile in PROFILES else None
    )
    ep_size = int(args.ep_size or default_ep_size or 8)
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
                    "strategy": args.strategy,
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
    main()
