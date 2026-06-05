#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark NCCL user-buffer registration for VisualGen collectives.

Two benchmark tiers:

  Tier 1 — Collective micro-benchmark (no model weights needed).
    Measures latency of all-to-all and all-gather collectives that dominate
    Ulysses sequence-parallel step time, with and without NCCL_CUMEM_ENABLE.
    Run this tier to get numbers quickly on any multi-GPU node.

  Tier 2 — Full pipeline benchmark (model weights required).
    Runs end-to-end generation with trtllm VisualGen and records wall-clock
    step latency over N iterations.

Usage — collective micro-benchmark (recommended first):
    torchrun --nproc-per-node=2 bench_nccl_ub.py --tier micro

Usage — full pipeline (requires weights):
    torchrun --nproc-per-node=2 bench_nccl_ub.py \\
        --tier pipeline \\
        --flux-path /path/to/FLUX.1-dev \\
        --wan-path  /path/to/Wan2.1-T2V-1.3B-Diffusers

Results are written to bench_nccl_ub_results.json in the current directory
(only from rank 0).
"""

import argparse
import json
import os
import time
from typing import Dict, List

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _timed_iters(fn, warmup: int, iters: int) -> List[float]:
    """Run fn() warmup times, then time iters calls.  Returns latencies in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _stats(times: List[float]) -> Dict:
    import statistics
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "n": len(times),
    }


# ---------------------------------------------------------------------------
# Tier 1: Collective micro-benchmark
# ---------------------------------------------------------------------------

def _bench_collective_one(
    group: dist.ProcessGroup,
    world_size: int,
    shape: tuple,
    dtype: torch.dtype,
    collective: str,
    warmup: int,
    iters: int,
) -> Dict:
    """Benchmark a single collective op on tensors of given shape."""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    tensor = torch.randn(*shape, dtype=dtype, device=device)

    if collective == "all_to_all":
        seq_dim_size = shape[1]  # [B, S/U, H/U, D]
        assert seq_dim_size % world_size == 0
        out = torch.empty_like(tensor)

        def fn():
            dist.all_to_all_single(out, tensor, group=group)

    elif collective == "all_gather":
        out = torch.empty(
            shape[0], shape[1] * world_size, *shape[2:], dtype=dtype, device=device
        )

        def fn():
            dist.all_gather_into_tensor(out, tensor, group=group)

    elif collective == "all_reduce":
        t = tensor.clone()

        def fn():
            dist.all_reduce(t, group=group)

    else:
        raise ValueError(f"Unknown collective: {collective}")

    times = _timed_iters(fn, warmup=warmup, iters=iters)
    return _stats(times)


def run_micro_benchmark(args) -> Dict:
    """Tier-1: no model weights, pure collective timing."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    results = {}

    # Shapes representative of FLUX (seq=4096 after patch) and Wan-1.3B (seq=7680)
    configs = [
        # (name, B, S/world_size, H/world_size, D, collective)
        ("flux_all_to_all",    "all_to_all",   (1, 4096 // world_size, 24 // world_size, 128)),
        ("flux_all_gather",    "all_gather",   (1, 4096 // world_size, 24, 128)),
        ("wan_all_to_all",     "all_to_all",   (1, 7680 // world_size, 40 // world_size, 128)),
        ("wan_all_gather",     "all_gather",   (1, 7680 // world_size, 40, 128)),
        ("all_reduce_small",   "all_reduce",   (1, 512, 512)),
    ]

    for name, collective, shape in configs:
        stat = _bench_collective_one(
            group=dist.group.WORLD,
            world_size=world_size,
            shape=shape,
            dtype=torch.bfloat16,
            collective=collective,
            warmup=args.warmup,
            iters=args.iters,
        )
        results[name] = stat
        if rank == 0:
            print(f"  {name:30s}  mean={stat['mean_ms']:.3f} ms  median={stat['median_ms']:.3f} ms")

    return results


# ---------------------------------------------------------------------------
# Tier 2: Full pipeline benchmark
# ---------------------------------------------------------------------------

def _bench_pipeline_one(model_path: str, config_path: str, prompt: str, n_steps: int,
                         warmup: int, iters: int, height: int, width: int) -> Dict:
    """Run trtllm VisualGen for n_steps, measure wall-clock per-image latency."""
    from tensorrt_llm import VisualGen, VisualGenArgs

    args = VisualGenArgs.from_yaml(config_path)
    args.model = model_path

    vg = VisualGen(args)

    # Warmup
    for _ in range(warmup):
        vg.generate(prompt=prompt, height=height, width=width, num_inference_steps=n_steps)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        vg.generate(prompt=prompt, height=height, width=width, num_inference_steps=n_steps)
        times.append((time.perf_counter() - t0) * 1000.0)

    vg.shutdown()
    return _stats(times)


def run_pipeline_benchmark(args) -> Dict:
    """Tier-2: end-to-end generation latency."""
    results = {}
    prompt = "A photo of a cat sitting on a red velvet throne"

    benchmarks = []
    if args.flux_path:
        benchmarks.append(
            ("flux_baseline",  args.flux_path,
             "examples/visual_gen/serve/configs/flux1.yml",     1024, 1024)
        )
        benchmarks.append(
            ("flux_nccl_ub",   args.flux_path,
             "examples/visual_gen/serve/configs/flux1_nccl_ub.yml", 1024, 1024)
        )
    if args.wan_path:
        benchmarks.append(
            ("wan_baseline",   args.wan_path,
             "examples/visual_gen/serve/configs/wan21.yml",     480,  832)
        )
        benchmarks.append(
            ("wan_nccl_ub",    args.wan_path,
             "examples/visual_gen/serve/configs/wan21_nccl_ub.yml",  480,  832)
        )

    for name, model_path, config_path, h, w in benchmarks:
        if _is_rank0():
            print(f"\nBenchmarking {name}  ({model_path}) ...")
        stat = _bench_pipeline_one(
            model_path=model_path,
            config_path=config_path,
            prompt=prompt,
            n_steps=args.steps,
            warmup=args.warmup,
            iters=args.iters,
            height=h,
            width=w,
        )
        results[name] = stat
        if _is_rank0():
            print(f"  {name:20s}  mean={stat['mean_ms']:.1f} ms  ({stat['n']} iters)")

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(results: Dict, tier: str):
    print(f"\n{'='*70}")
    print(f"NCCL User-Buffer Registration Benchmark — {tier}")
    print(f"{'='*70}")
    print(f"{'Key':<30}  {'Mean ms':>10}  {'Median ms':>10}  {'Min ms':>8}")
    print(f"{'-'*30}  {'-'*10}  {'-'*10}  {'-'*8}")
    for k, v in results.items():
        print(f"{k:<30}  {v['mean_ms']:>10.3f}  {v['median_ms']:>10.3f}  {v['min_ms']:>8.3f}")
    print()

    # For pipeline tier: compute speedup
    if tier == "pipeline":
        for prefix in ("flux", "wan"):
            base = results.get(f"{prefix}_baseline")
            ub = results.get(f"{prefix}_nccl_ub")
            if base and ub:
                speedup = base["mean_ms"] / ub["mean_ms"]
                delta_ms = base["mean_ms"] - ub["mean_ms"]
                print(f"{prefix.upper()} speedup: {speedup:.3f}x  ({delta_ms:+.1f} ms per image)")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", choices=["micro", "pipeline"], default="micro",
                        help="micro: collective microbenchmark (no weights); "
                             "pipeline: full end-to-end (requires --flux-path or --wan-path)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=10, help="Timed iterations")
    parser.add_argument("--steps", type=int, default=20,
                        help="Denoising steps for pipeline tier")
    parser.add_argument("--flux-path", type=str, default="",
                        help="Path to FLUX.1-dev checkpoint dir")
    parser.add_argument("--wan-path", type=str, default="",
                        help="Path to Wan2.1-T2V-1.3B-Diffusers checkpoint dir")
    parser.add_argument("--out", type=str, default="bench_nccl_ub_results.json",
                        help="Output JSON path (rank-0 only)")
    parser.add_argument("--nccl-cumem", action="store_true", default=False,
                        help="Force NCCL_CUMEM_ENABLE=1 for this process "
                             "(micro tier: test WITH registration; "
                             "use two separate torchrun invocations to compare)")
    args = parser.parse_args()

    # CUMEM must be set before init_process_group
    if args.nccl_cumem:
        os.environ["NCCL_CUMEM_ENABLE"] = "1"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        nccl_cumem = os.environ.get("NCCL_CUMEM_ENABLE", "0")
        print(f"\nNCCL UB Benchmark  tier={args.tier}  "
              f"world_size={world_size}  NCCL_CUMEM_ENABLE={nccl_cumem}")
        print(f"warmup={args.warmup}  iters={args.iters}\n")

    if args.tier == "micro":
        results = run_micro_benchmark(args)
    else:
        if not args.flux_path and not args.wan_path:
            parser.error("Pipeline tier requires at least one of --flux-path / --wan-path")
        results = run_pipeline_benchmark(args)

    if rank == 0:
        _print_summary(results, args.tier)
        payload = {
            "tier": args.tier,
            "world_size": world_size,
            "nccl_cumem_enable": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
            "warmup": args.warmup,
            "iters": args.iters,
            "results": results,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {args.out}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
