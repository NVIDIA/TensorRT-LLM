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

Launch (examples):

```bash
# Basic usage
python tests/microbenchmarks/bench_moe_comm.py --ep_size 8 --backend DEEPEP --profile deepseek_v3

# Show per-kernel breakdown and stats across iterations
python tests/microbenchmarks/bench_moe_comm.py --ep_size 8 --backend NVLINK_ONE_SIDED --kernel_breakdown --iter_stats

# With batch size sweeping
python tests/microbenchmarks/bench_moe_comm.py --ep_size 8 --backend NVLINK_ONE_SIDED -b 640 -e 2048 -f 2

# Run all supported strategies for the given profile
python tests/microbenchmarks/bench_moe_comm.py --ep_size 8 --profile deepseek_v3

```
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cloudpickle
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from torch.autograd import DeviceType

import tensorrt_llm as tllm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import CutlassFusedMoE, MoE
from tensorrt_llm._torch.modules.fused_moe.communication import Communication, CommunicationFactory
from tensorrt_llm._torch.modules.fused_moe.routing import DefaultMoeRoutingMethod
from tensorrt_llm._utils import local_mpi_rank, mpi_allgather, mpi_barrier, mpi_rank, mpi_world_size
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


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
    quant_algo: QuantAlgo


PROFILES: Dict[str, Profile] = {
    # DeepSeek-V3: hidden_size 7168, router_topk 8 (public config)
    "deepseek_v3": Profile(
        name="deepseek_v3",
        hidden_size=7168,
        top_k=8,
        # Previously: experts_per_rank=32 with recommended ep_size=8 => 256 total experts.
        num_experts=256,
        quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
    ),
    # Repo already references "gpt-oss" hidden_size=2880 in the MoE A2A unit test.
    "gpt_oss": Profile(
        name="gpt_oss",
        hidden_size=2880,
        top_k=4,
        num_experts=128,
        quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8,
    ),
}


def _maybe_warn_rank0(msg: str):
    if mpi_rank() == 0:
        print(msg, flush=True)


def _sync():
    torch.cuda.synchronize()
    mpi_barrier()


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
    quant_algo: QuantAlgo,
    backend: Communication,
    moe: Optional[MoE] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    # Hidden states: payload we want to communicate.
    hidden_states = torch.randn(local_num_tokens, hidden_size, dtype=act_dtype, device=device)
    # We keep scaling factors optional; most strategies can ignore it.
    hidden_states_sf = None
    # Post-quant communication: Quantize → Dispatch (mirrors ConfigurableMoE ordering),
    # using Cutlass' quantize_input() (outside the timed comm region).
    if quant_algo != QuantAlgo.NO_QUANT and backend.supports_post_quant_dispatch():
        hidden_states, hidden_states_sf = moe.quantize_input(hidden_states, post_quant_comm=True)
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
    quant_config: Optional[QuantConfig],
) -> ModelConfig:
    # Keep it minimal: just enough fields for CommunicationFactory.
    return ModelConfig(
        pretrained_config=_DummyPretrainedConfig(hidden_size=hidden_size, torch_dtype=act_dtype),
        mapping=mapping,
        quant_config=(quant_config or QuantConfig()),
        max_num_tokens=int(max_num_tokens_per_rank),
        moe_max_num_tokens=int(max_num_tokens_per_rank),
        use_cuda_graph=False,
        use_low_precision_moe_combine=False,
    )


def _time_dispatch_and_combine(
    backend: Communication,
    *,
    hidden_states: torch.Tensor,
    hidden_states_sf: Optional[torch.Tensor],
    token_selected_slots: torch.Tensor,
    token_final_scales: Optional[torch.Tensor],
    all_rank_num_tokens: List[int],
    hidden_size: int,
    warmup: int,
    iters: int,
    flush_l2: bool = True,
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """Time dispatch and combine using Kineto (torch.profiler with CUPTI).

    Returns:
        dispatch_times_us: Per-iteration dispatch GPU times in microseconds
        combine_times_us: Per-iteration combine GPU times in microseconds
        detailed_stats: Dict containing per-kernel timing breakdown
    """
    device = hidden_states.device

    # Prepare dispatch once (excluded from timing)
    _ = backend.prepare_dispatch(token_selected_slots, all_rank_num_tokens, None)

    # L2 cache flushing buffer
    l2_buffer = None
    if flush_l2:
        l2_size = torch.cuda.get_device_properties(device).L2_cache_size
        # Use 2x L2 size to ensure complete flush
        l2_flush_size = (l2_size * 2) // 4  # Size in int32 elements
        l2_buffer = torch.empty(l2_flush_size, dtype=torch.int32, device=device)

    # Warmup iterations (not profiled)
    for _ in range(warmup):
        if l2_buffer is not None:
            l2_buffer.zero_()
        recv_hidden_states, _, _, _ = backend.dispatch(
            hidden_states,
            hidden_states_sf,
            token_selected_slots,
            token_final_scales,
            all_rank_num_tokens,
        )
        shape = list(recv_hidden_states.shape)
        shape[-1] = hidden_size
        recv_hidden_states_moe = torch.empty(
            tuple(shape), dtype=torch.bfloat16, device=recv_hidden_states.device
        )
        _ = backend.combine(
            recv_hidden_states_moe, all_rank_max_num_tokens=max(all_rank_num_tokens)
        )

    # Profile with Kineto
    with torch.profiler.profile(
        # Include CPU so `record_function("dispatch_iter..."/"combine_iter...")` ranges
        # appear in key_averages() / events(). Without CPU activity those ranges are
        # missing, causing dispatch/combine attribution to fail.
        activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        _sync()
        for _ in range(iters):
            # L2 cache flushing
            if l2_buffer is not None:
                l2_buffer.zero_()

            # Mark dispatch operation for aggregated timing
            with torch.profiler.record_function("dispatch"):
                recv_hidden_states, _, _, _ = backend.dispatch(
                    hidden_states,
                    hidden_states_sf,
                    token_selected_slots,
                    token_final_scales,
                    all_rank_num_tokens,
                )

            # Simulate MoE computation output
            shape = list(recv_hidden_states.shape)
            shape[-1] = hidden_size
            recv_hidden_states_moe = torch.empty(
                tuple(shape), dtype=torch.bfloat16, device=recv_hidden_states.device
            )

            # Mark combine operation for aggregated timing
            with torch.profiler.record_function("combine"):
                _ = backend.combine(
                    recv_hidden_states_moe, all_rank_max_num_tokens=max(all_rank_num_tokens)
                )

    _sync()

    # if mpi_rank() == 0:
    #     print("########################################################")
    #     print(prof.key_averages())
    #     print("########################################################")

    # ------------------------------------------------------------------
    # Categorize GPU kernels by enclosing record_function scope
    # ("dispatch" or "combine").
    #
    # Each record_function marker produces both a CPU event and a CUDA
    # range event.  We collect the GPU-side "dispatch"/"combine" ranges
    # and check whether each GPU kernel's start timestamp falls inside
    # one of them (GPU time-range containment).
    # ------------------------------------------------------------------
    events_list = list(prof.events())
    # if mpi_rank() == 0:
    #     print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #     for evt in events_list:
    #         print(evt)
    #     print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    def _is_gpu_event(evt) -> bool:
        return getattr(evt, "device_type", None) == DeviceType.CUDA

    # Step 1: Collect GPU time ranges of "dispatch"/"combine" CUDA ranges.
    gpu_dispatch_intervals: List[Tuple[int, int]] = []
    gpu_combine_intervals: List[Tuple[int, int]] = []

    for evt in events_list:
        if not _is_gpu_event(evt) or evt.name not in ("dispatch", "combine"):
            continue
        tr = getattr(evt, "time_range", None)
        if tr is None:
            continue
        assert tr.end > tr.start
        (gpu_dispatch_intervals if evt.name == "dispatch" else gpu_combine_intervals).append(
            (tr.start, tr.end)
        )
    gpu_dispatch_intervals.sort()
    gpu_combine_intervals.sort()

    # Step 2: Scope resolver (GPU events only) ---------------------------------
    def _find_scope(evt) -> Optional[str]:
        """Return 'dispatch', 'combine', or None based on GPU time-range containment."""
        tr = getattr(evt, "time_range", None)
        if tr is None:
            return None
        t = tr.start
        for s, e in gpu_dispatch_intervals:
            if s <= t <= e:
                return "dispatch"
        for s, e in gpu_combine_intervals:
            if s <= t <= e:
                return "combine"
        return None

    # Step 3: Iterate events and bucket by scope ----------------------------
    dispatch_kernel_times: Dict[str, List[float]] = {}
    combine_kernel_times: Dict[str, List[float]] = {}
    other_kernel_times: Dict[str, List[float]] = {}

    for evt in events_list:
        if not _is_gpu_event(evt):
            continue
        if evt.device_time <= 0:
            continue
        if evt.name in ("dispatch", "combine"):
            continue  # skip record_function range markers

        scope = _find_scope(evt)
        if scope == "dispatch":
            dispatch_kernel_times.setdefault(evt.name, []).append(evt.device_time)
        elif scope == "combine":
            combine_kernel_times.setdefault(evt.name, []).append(evt.device_time)
        else:
            other_kernel_times.setdefault(evt.name, []).append(evt.device_time)

    # Step 4: Build per-kernel stats ----------------------------------------
    def _build_kernel_list(kernel_times: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        result = []
        for name, times in kernel_times.items():
            result.append(
                {
                    "name": name,
                    "count": len(times),
                    "_times": times,  # raw per-iteration times, gathered across ranks later
                }
            )
        return result

    dispatch_kernels = _build_kernel_list(dispatch_kernel_times)
    combine_kernels = _build_kernel_list(combine_kernel_times)
    other_kernels = _build_kernel_list(other_kernel_times)

    # Step 5: Collect per-iteration dispatch/combine times (us) ---------------
    # Use the CUDA-side "dispatch"/"combine" range events (device_type=CUDA)
    # for direct GPU time measurement.
    dispatch_times_us: List[float] = []
    combine_times_us: List[float] = []
    for evt in events_list:
        if not _is_gpu_event(evt):
            continue
        if evt.name == "dispatch":
            dispatch_times_us.append(evt.device_time)
        elif evt.name == "combine":
            combine_times_us.append(evt.device_time)

    # Sort each category by mean time descending
    dispatch_kernels.sort(
        key=lambda x: sum(x["_times"]) / len(x["_times"]) if x["_times"] else 0, reverse=True
    )
    combine_kernels.sort(
        key=lambda x: sum(x["_times"]) / len(x["_times"]) if x["_times"] else 0, reverse=True
    )
    other_kernels.sort(
        key=lambda x: sum(x["_times"]) / len(x["_times"]) if x["_times"] else 0, reverse=True
    )

    detailed_stats = {
        "dispatch_kernels": dispatch_kernels,
        "combine_kernels": combine_kernels,
        "other_kernels": other_kernels,
    }

    return dispatch_times_us, combine_times_us, detailed_stats


def _compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics over a list of values."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / n
    return {
        "mean": mean,
        "median": s[n // 2],
        "stdev": variance**0.5,
        "min": s[0],
        "max": s[-1],
    }


def _gather_per_rank(times_us: List[float], iter_stats: bool = False) -> Dict[str, Any]:
    """Allgather per-iteration times from each rank, return per-rank results.

    If iter_stats=True, return full stats (mean/median/stdev/min/max).
    If iter_stats=False, return just the mean.
    """
    all_times = mpi_allgather(times_us)
    if iter_stats:
        return {f"rank{i}": _compute_stats(t) for i, t in enumerate(all_times)}
    return {f"rank{i}": (sum(t) / len(t) if t else 0.0) for i, t in enumerate(all_times)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified MoE communication microbenchmark (MPI).")
    parser.add_argument(
        "--ep_size",
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
            "DEEPEP",
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
    parser.add_argument("--hidden_size", type=int, default=None, help="Custom hidden size.")
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
    parser.add_argument("--top_k", type=int, default=None, help="Custom router top-k.")
    parser.add_argument(
        "--num_experts",
        type=int,
        default=None,
        help="Total number of experts.",
    )
    parser.add_argument(
        "--quant",
        type=lambda s: str(s).upper(),
        default=None,
        choices=[q.name for q in QuantAlgo],
        help="Override quantization algo (defaults to profile.quant_algo).",
    )
    parser.add_argument("--iters", type=int, default=200, help="Timed iterations.")
    parser.add_argument(
        "--warmup", type=int, default=20, help="Warmup iterations. (Currently ignored.)"
    )
    parser.add_argument(
        "--max_num_tokens_per_rank",
        type=int,
        default=None,
        help="Max tokens per rank for workspace allocation (defaults to max scanned local_batch_size).",
    )
    parser.add_argument(
        "--kernel_breakdown",
        action="store_true",
        help="Show per-kernel timing breakdown.",
    )
    parser.add_argument(
        "--iter_stats",
        action="store_true",
        help="Show detailed per-iteration stats (mean/median/stdev/min/max) instead of just mean.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to write JSON report file (default: None, stdout only).",
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


def _resolve_profile_args(args: argparse.Namespace) -> Tuple[int, int, int, int, QuantAlgo]:
    """Returns (hidden_size, local_num_tokens, top_k, num_experts_total, quant_algo)."""
    local_num_tokens = _iter_local_batch_sizes(args)[0]

    # If a profile is provided, it supplies defaults; any explicit CLI values override.
    if args.profile is not None:
        prof = PROFILES[args.profile]
        hidden_size = int(args.hidden_size or prof.hidden_size)
        top_k = int(args.top_k or prof.top_k)
        num_experts_total = int(args.num_experts or prof.num_experts)
        quant_algo = prof.quant_algo
        return hidden_size, local_num_tokens, top_k, num_experts_total, quant_algo

    # No profile: all fields must be provided explicitly.
    if args.hidden_size is None or args.top_k is None or args.num_experts is None:
        raise ValueError(
            "No --profile specified; must provide --hidden_size, --top_k, --num_experts."
        )
    hidden_size = int(args.hidden_size)
    top_k = int(args.top_k)
    num_experts_total = int(args.num_experts)
    return hidden_size, local_num_tokens, top_k, num_experts_total, QuantAlgo.NO_QUANT


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

    hidden_size, _, top_k, num_experts_total, profile_quant_algo = _resolve_profile_args(args)
    local_batch_sizes = _iter_local_batch_sizes(args)
    act_dtype = torch.bfloat16
    quant_algo = QuantAlgo[args.quant] if args.quant is not None else profile_quant_algo
    quant_config = (
        QuantConfig(quant_algo=None)
        if quant_algo == QuantAlgo.NO_QUANT
        else QuantConfig(quant_algo=quant_algo)
    )

    max_scanned_tokens = max(local_batch_sizes)
    max_num_tokens_per_rank = int(args.max_num_tokens_per_rank or max_scanned_tokens)
    if max_num_tokens_per_rank < max_scanned_tokens:
        raise ValueError(
            "--max_num_tokens_per_rank must be >= the maximum scanned local_batch_size "
            "(from -b/--minbatch..-e/--maxbatch)."
        )

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
                    "quant_algo": quant_algo.name,
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
        ["ALLGATHER", "NVLINK_ONE_SIDED", "NVLINK_TWO_SIDED", "DEEPEP", "DEEPEPLOWLATENCY"]
        if args.backend is None
        else [args.backend]
    )

    all_results: List[Dict[str, Any]] = []

    for backend_name in backends:
        try:
            model_config = _create_model_config(
                mapping=mapping,
                hidden_size=hidden_size,
                act_dtype=act_dtype,
                max_num_tokens_per_rank=max_num_tokens_per_rank,
                quant_config=quant_config,
            )

            backend = CommunicationFactory._create_forced_method(  # pylint: disable=protected-access
                backend_name,
                model_config,
                num_experts_total,
                num_slots,
                top_k,
                experts_per_rank,
                payload_in_workspace=False,
                alltoall_result_do_sum=True,
            )

            if backend is None:
                _maybe_warn_rank0(
                    f"[bench_moe_comm] Skipping {backend_name}: factory returned None."
                )
                mpi_barrier()
                continue
        except Exception as e:
            _maybe_warn_rank0(f"[bench_moe_comm] Skipping {backend_name}: {type(e).__name__}: {e}")
            mpi_barrier()
            continue

        # Post-quant communication: Quantize → Dispatch (mirrors ConfigurableMoE ordering),
        # using Cutlass' quantize_input() (outside the timed comm region).
        moe = None
        if quant_algo != QuantAlgo.NO_QUANT and backend.supports_post_quant_dispatch():
            routing_method = DefaultMoeRoutingMethod(top_k=top_k)
            moe = CutlassFusedMoE(
                routing_method=routing_method,
                num_experts=num_experts_total,
                hidden_size=hidden_size,
                intermediate_size=hidden_size * 4,
                dtype=torch.bfloat16,
                reduce_results=False,
                model_config=model_config,
                init_load_balancer=False,
                without_comm=True,
            )
            # Ensure quantization params (e.g., NVFP4 global scale) live on CUDA.
            moe = moe.to(device)

        for local_num_tokens in local_batch_sizes:
            all_rank_num_tokens = mpi_allgather(int(local_num_tokens))
            try:
                if not backend.is_workload_feasible(all_rank_num_tokens, num_chunks=1):
                    _maybe_warn_rank0(
                        f"[bench_moe_comm] Skipping {backend_name} @ local_batch_size={local_num_tokens}: workload not feasible."
                    )
                    mpi_barrier()
                    continue
            except Exception as e:
                _maybe_warn_rank0(
                    f"[bench_moe_comm] Skipping {backend_name} @ local_batch_size={local_num_tokens}: feasibility check failed: {e}"
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
                    quant_algo,
                    backend,
                    moe,
                )
            )

            # Time dispatch and combine with Kineto
            dispatch_times_us, combine_times_us, detailed_stats = _time_dispatch_and_combine(
                backend,
                hidden_states=hidden_states,
                hidden_states_sf=hidden_states_sf,
                token_selected_slots=token_selected_slots,
                token_final_scales=token_final_scales,
                all_rank_num_tokens=all_rank_num_tokens,
                hidden_size=hidden_size,
                warmup=int(args.warmup),
                iters=int(args.iters),
                flush_l2=True,
            )

            iter_stats = bool(args.iter_stats)
            dispatch_stats = _gather_per_rank(dispatch_times_us, iter_stats=iter_stats)
            combine_stats = _gather_per_rank(combine_times_us, iter_stats=iter_stats)

            # Prepare output
            output = {
                "backend": backend_name,
                "local_batch_size": int(local_num_tokens),
                "dispatch_us": dispatch_stats,
                "combine_us": combine_stats,
            }

            # Add kernel breakdown if requested and available
            if args.kernel_breakdown and detailed_stats is not None:
                for category in ("dispatch_kernels", "combine_kernels", "other_kernels"):
                    kernels = detailed_stats.get(category, [])
                    for kernel in kernels:
                        kernel["per_rank"] = _gather_per_rank(
                            kernel.pop("_times", []), iter_stats=iter_stats
                        )
                    output[category] = kernels

            if rank == 0:
                print(json.dumps(output, indent=2), flush=True)
                all_results.append(output)

            mpi_barrier()

    # Write JSON report if requested
    if rank == 0 and args.output_file and all_results:
        with open(args.output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Report written to {args.output_file}", flush=True)

    return


def _spawn_worker_main(args_blob: bytes) -> List[Dict[str, Any]]:
    """Worker entrypoint for MPIPoolExecutor.

    Note: MPIPoolExecutor typically serializes callables by name; for scripts executed
    as __main__, we set MPI's pickler to cloudpickle in the parent so this function is
    available to workers (same pattern as our unit tests).
    """
    args = pickle.loads(args_blob)
    # In spawned workers, we are already inside an MPI world of size == ep_size.
    try:
        _run_benchmark_worker_under_current_mpi(args)
    except Exception as e:
        # Make worker-side stack trace visible at the parent.
        rank = mpi_rank()
        size = mpi_world_size()
        msg = (
            "[bench_moe_comm worker] uncaught exception:\n"
            f"rank={rank}/{size} local_rank={local_mpi_rank()} pid={os.getpid()}\n"
            "Worker traceback:\n"
            f"{traceback.format_exc()}"
        )
        try:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        except Exception:
            pass
        raise RuntimeError(msg) from e
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
        raise ValueError("--ep_size must be > 0")

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
    # Opt-in to DeepEP backends by default. This does not override an explicit user setting.
    os.environ.setdefault("TRTLLM_CAN_USE_DEEP_EP", "1")
    main()
