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

# ruff: noqa: E402, E501
"""Standalone MEGAMOE_CUTEDSL MoE-kernel perf microbench (MPI, EP/DEP).

Drives the ``MEGAMOE_CUTEDSL`` backend through the public ``create_moe`` ->
``forward`` path with synthetic NVFP4 weights + uniform-random routing, and
times the fused dispatch + fc1 + fc2 + combine MoE forward.

Timing: the default times the eager forward, which includes host kernel-launch
gaps and so overstates CUDA-graphed serving (most at low token counts).
``--cuda-graph`` captures and replays the forward for a kernel-bound number
(barrier between replays to re-align EP ranks, drop iter-0); keep
``--iters-per-graph`` >= 4.

Launch (8 GPUs, EP8 / attention-DP; add --cuda-graph for kernel-bound perf):

```bash
mpirun --allow-run-as-root --oversubscribe --bind-to none --map-by slot -np 8 \
  python tests/microbenchmarks/bench_megamoe_cutedsl.py --cuda-graph \
    --tokens-per-rank 1024 2048 4096 --combine-format fp8 \
    --warmup 16 --iters 30 --output-file bench_mm.json
```

Perf knobs (forwarded to the backend as env vars):
- ``--combine-format {bf16,fp8,fp4}`` -> ``MEGAMOE_COMBINE_FORMAT`` (bf16 / 32e4m3xe8m0 / 16e2m1xbf16)
- ``--autotune``                      -> ``MEGAMOE_AUTOTUNE=1`` (sweep tactics vs the heuristic; forces a single bucket)
- ``--cuda-graph``                    -> kernel-bound timing (CUDA-graph replay; slowest EP rank, min over iters); default is eager forward latency

NOTE: the kernel is launched via cute.compile JIT, which compiles once per
``max_tokens_per_rank`` bucket and has NO cross-process cache. Use ``--warmup``
>= 16 so the compile is excluded from the timed window (otherwise throughput is
understated). A fresh backend is built per ``--tokens-per-rank`` value; if a
high token count OOMs, run that point in its own process.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tests" / "unittest"))

from _torch.modules.moe.moe_test_utils import MoeBackendType
from _torch.modules.moe.quantize_utils import get_test_quant_params
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod, create_moe
from tensorrt_llm._torch.modules.fused_moe.interface import MoEWeightLoadingMode
from tensorrt_llm._utils import mpi_rank, mpi_world_size
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

_COMBINE = {"bf16": "bf16", "fp8": "32e4m3xe8m0", "fp4": "16e2m1xbf16"}


def _ensure_dist(rank: int, world_size: int) -> None:
    if dist.is_initialized():
        return
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29561")
    # Global rank drives EP/Mapping; the CUDA device is the LOCAL index (robust
    # to multi-node / one-GPU-per-process launchers, where global rank != device).
    local_rank = rank % max(1, torch.cuda.device_count())
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _time_cuda_graph(fwd, args, ws: int) -> list:
    """Time the forward via CUDA-graph replay (kernel-bound; see module docstring)."""
    ipg = args.iters_per_graph
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(max(args.warmup, 3)):
            fwd()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    if dist.is_initialized() and ws > 1:
        dist.barrier()  # align EP ranks before the capture executes the kernel
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(ipg):
            fwd()
    graph.replay()  # prime: absorb first-replay setup skew (untimed)
    torch.cuda.synchronize()
    per_iter = []
    for _ in range(args.iters):
        if dist.is_initialized() and ws > 1:
            dist.barrier()  # re-align ranks; OUTSIDE the timed record window
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        per_iter.append(start.elapsed_time(end) / ipg)
    return per_iter


def _bench_one(tpr: int, mapping: Mapping, args, dtype) -> dict:
    """Build, warm up, and time a fresh MEGAMOE_CUTEDSL backend for ``tpr`` per-rank tokens."""
    ws = mapping.world_size
    num_experts, top_k = args.num_experts, args.top_k
    hidden, intermediate = args.hidden, args.intermediate
    routing = RenormalizeMoeRoutingMethod(top_k=top_k)
    # Deterministic per-rank inputs (reproducible run-to-run; balanced routing
    # in expectation from the random logits under RenormalizeMoeRoutingMethod).
    torch.manual_seed(1234 + mapping.rank)
    x = torch.randn((tpr, hidden), dtype=dtype, device="cuda")
    router_logits = torch.randn((tpr, num_experts), dtype=dtype, device="cuda")
    all_rank_num_tokens = [tpr] * ws

    backend_type = MoeBackendType("MEGAMOE_CUTEDSL")
    qcls, qconfig, qkwargs = get_test_quant_params(QuantAlgo.NVFP4, x, backend_type)
    qkwargs.pop("ref_cls", None)
    qutil = qcls(
        num_experts=num_experts,
        dtype=dtype,
        intermediate_size=intermediate,
        hidden_size=hidden,
        quant_config=qconfig,
        num_local_experts=num_experts // mapping.moe_ep_size,
    )
    pc = PretrainedConfig()
    pc.num_experts = num_experts
    pc.hidden_size = hidden
    pc.intermediate_size = intermediate
    pc.torch_dtype = dtype
    mcfg = ModelConfig(
        pretrained_config=pc,
        mapping=mapping,
        quant_config=qconfig,
        moe_backend="MEGAMOE_CUTEDSL",
        moe_disable_finalize_fusion=False,
        moe_load_balancer=None,
        max_num_tokens=tpr,
    )

    with create_moe(
        routing_method=routing,
        reduce_results=True,
        model_config=mcfg,
        weight_loading_mode=MoEWeightLoadingMode.VANILLA,
    ) as moe:
        weights = qutil.create_weights(**qkwargs)
        moe.load_weights([weights])
        moe.post_load_weights()
        moe.cuda(f"cuda:{torch.cuda.current_device()}")

        def _fwd():
            return moe.forward(x, router_logits, all_rank_num_tokens=all_rank_num_tokens)

        with torch.inference_mode():
            if args.cuda_graph:
                per_iter = _time_cuda_graph(_fwd, args, ws)
            else:
                for _ in range(args.warmup):
                    _fwd()
                torch.cuda.synchronize()
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
                for i in range(args.iters):
                    starts[i].record()
                    _fwd()
                    ends[i].record()
                torch.cuda.synchronize()
                per_iter = [starts[i].elapsed_time(ends[i]) for i in range(args.iters)]
            # Report the SLOWEST EP rank per iteration (the lockstep critical
            # path): an in-kernel all-to-all MoE completes only when every rank
            # does, so rank-0-only times understate latency. All-reduce MAX over
            # the EP group (outside the timed window) before taking statistics.
            if dist.is_initialized() and ws > 1:
                _t = torch.tensor(per_iter, dtype=torch.float64, device="cuda")
                dist.all_reduce(_t, op=dist.ReduceOp.MAX)
                per_iter = _t.cpu().tolist()
            ms = sorted(per_iter)

    mean_ms = sum(ms) / len(ms)
    total = tpr * ws
    res = {
        "tokens_per_rank": tpr,
        "total_tokens": total,
        "ms_mean": round(mean_ms, 4),
        "ms_min": round(ms[0], 4),
        "ms_p50": round(ms[len(ms) // 2], 4),
        "ms_p90": round(ms[min(int(len(ms) * 0.9), len(ms) - 1)], 4),
        "tokens_per_s": round(total / (mean_ms / 1e3), 1),
        "combine_format": args.combine_format,
        "autotune": bool(args.autotune),
        "timing": "cuda_graph" if args.cuda_graph else "eager_forward",
        "iters_per_graph": args.iters_per_graph if args.cuda_graph else None,
    }
    del weights
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--tokens-per-rank",
        type=int,
        nargs="+",
        default=[4096],
        help="per-rank token counts to sweep (TOTAL = value * world_size)",
    )
    ap.add_argument("--combine-format", choices=list(_COMBINE), default="bf16")
    ap.add_argument(
        "--autotune",
        action="store_true",
        help="MEGAMOE_AUTOTUNE=1 (sweep tactics; forces a single bucket)",
    )
    ap.add_argument(
        "--cuda-graph",
        action="store_true",
        help="kernel-bound timing via CUDA-graph replay (slowest EP rank, min "
        "over iters); default eager forward latency overstates serving. Keep "
        "--iters-per-graph >= 4.",
    )
    ap.add_argument(
        "--iters-per-graph",
        type=int,
        default=4,
        help="forwards captured per CUDA graph (--cuda-graph only); keep >= 4.",
    )
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--intermediate", type=int, default=2048)
    ap.add_argument("--output-file", default=None)
    args = ap.parse_args()
    if args.iters < 1 or args.iters_per_graph < 1:
        raise SystemExit("--iters and --iters-per-graph must be >= 1")

    import os

    os.environ["MEGAMOE_COMBINE_FORMAT"] = _COMBINE[args.combine_format]
    if args.autotune:
        os.environ["MEGAMOE_AUTOTUNE"] = "1"

    rank, world_size = mpi_rank(), mpi_world_size()
    if world_size < 2:
        raise SystemExit(
            "Launch under mpirun with -np >= 2 (EP needs >1 rank), "
            "e.g. mpirun -np 8 python tests/microbenchmarks/bench_megamoe_cutedsl.py ..."
        )
    # DEP: attention DP, MoE EP -- the DSv4-Pro serving topology.
    mapping = Mapping(
        world_size=world_size,
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    mapping.rank = rank
    _ensure_dist(rank, world_size)
    if args.num_experts % world_size != 0:
        raise SystemExit(
            f"--num-experts ({args.num_experts}) must be divisible by world_size ({world_size})"
        )

    dtype = torch.bfloat16
    results = []
    for tpr in args.tokens_per_rank:
        results.append(_bench_one(tpr, mapping, args, dtype))

    if rank == 0:
        _timing = f"cuda_graph(ipg={args.iters_per_graph})" if args.cuda_graph else "eager_forward"
        cfg = (
            f"backend=MEGAMOE_CUTEDSL combine={args.combine_format} "
            f"timing={_timing} "
            f"autotune={bool(args.autotune)} EP{world_size} "
            f"e{args.num_experts}/k{args.top_k}/h{args.hidden}/i{args.intermediate} "
            f"warmup={args.warmup} iters={args.iters}"
        )
        print(f"\n=== MegaMoE-CuteDSL kernel microbench ({cfg}) ===")
        print(
            f"{'tok/rank':>9} {'total':>8} {'ms_mean':>9} {'ms_p50':>8} {'ms_p90':>8} {'tok/s':>12}"
        )
        for r in results:
            print(
                f"{r['tokens_per_rank']:>9} {r['total_tokens']:>8} {r['ms_mean']:>9} "
                f"{r['ms_p50']:>8} {r['ms_p90']:>8} {r['tokens_per_s']:>12}"
            )
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump({"config": cfg, "results": results}, f, indent=2)
            print(f"[written] {args.output_file}")


if __name__ == "__main__":
    main()
