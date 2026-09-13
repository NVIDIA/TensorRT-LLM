#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Benchmark ring-attention P2P transports and VisualGen collectives.

Two benchmark tiers:

  Tier 1 — Collective + ring P2P micro-benchmark (no model weights needed).
    Measures latency of:
      (a) all-to-all / all-gather collectives (Ulysses sequence parallelism)
      (b) ring P2P KV-block exchange (ring attention) across transports:
          NCCL ncclSend/Recv, UserBuffers (UB), and NIXL.
    Run this tier to get numbers quickly on any multi-GPU node.

  Tier 2 — Full pipeline benchmark (model weights required).
    Runs end-to-end generation with trtllm VisualGen and records wall-clock
    step latency over N iterations.

Usage — ring P2P micro-benchmark (all transports, recommended first):
    torchrun --nproc-per-node=4 bench_ring_transport.py --tier micro --backend all

Usage — NIXL ring transport only:
    TRTLLM_RING_TRANSPORT=nixl torchrun --nproc-per-node=4 bench_ring_transport.py \\
        --tier micro --backend nixl

Usage — full pipeline (requires weights):
    torchrun --nproc-per-node=4 bench_ring_transport.py \\
        --tier pipeline \\
        --flux-path /path/to/FLUX.1-dev \\
        --wan-path  /path/to/Wan2.1-T2V-1.3B-Diffusers

Results are written to bench_ring_transport_results.json (rank-0 only).

Side-by-side comparison report is written to bench_ring_transport_report.md.
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
        out = torch.empty(shape[0], shape[1] * world_size, *shape[2:], dtype=dtype, device=device)

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


def _bench_ub_alltoall_one(
    group: dist.ProcessGroup,
    world_size: int,
    shape: tuple,
    dtype: torch.dtype,
    collective: str,
    warmup: int,
    iters: int,
) -> Dict:
    """Benchmark a single collective using TRT-LLM UserBuffers (UB)."""
    from tensorrt_llm._torch.visual_gen.nccl_ub_reg import UBAllToAll, _ub_available

    if not _ub_available():
        return {"error": "UB not available on this system"}

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    tensor = torch.randn(*shape, dtype=dtype, device=device)
    nelems = tensor.numel()
    ub_a2a = UBAllToAll(max_elements=nelems * 2, dtype=dtype)

    if collective == "all_to_all":

        def fn():
            ub_a2a(tensor, scatter_dim=2, gather_dim=1, process_group=group)
    elif collective == "all_reduce":
        # UB allreduce is handled by the plugin layer; approximate with alltoall for now.

        def fn():
            dist.all_reduce(tensor, group=group)
    else:
        return {"error": f"UB micro-bench not implemented for {collective}"}

    times = _timed_iters(fn, warmup=warmup, iters=iters)
    return _stats(times)


# ---------------------------------------------------------------------------
# Ring P2P micro-benchmark  (NCCL / UB / NIXL)
# ---------------------------------------------------------------------------


def _bench_ring_p2p_nccl(
    group: dist.ProcessGroup, world_size: int, slot_bytes: int, warmup: int, iters: int
) -> Dict:
    """Benchmark one ring P2P step using NCCL ncclSend/ncclRecv (via torch.distributed)."""
    rank = dist.get_rank(group=group)
    send_rank = dist.get_global_rank(group, (rank + 1) % world_size)
    recv_rank = dist.get_global_rank(group, (rank - 1) % world_size)
    send_first = rank % 2 == 0
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    send_buf = torch.zeros(slot_bytes // 2, dtype=torch.bfloat16, device=device)
    recv_buf = torch.zeros_like(send_buf)

    def fn():
        if send_first:
            ops = [
                dist.P2POp(dist.isend, send_buf, send_rank, group=group),
                dist.P2POp(dist.irecv, recv_buf, recv_rank, group=group),
            ]
        else:
            ops = [
                dist.P2POp(dist.irecv, recv_buf, recv_rank, group=group),
                dist.P2POp(dist.isend, send_buf, send_rank, group=group),
            ]
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()

    times = _timed_iters(fn, warmup=warmup, iters=iters)
    return _stats(times)


def _bench_ring_p2p_nixl(
    group: dist.ProcessGroup, world_size: int, slot_bytes: int, warmup: int, iters: int
) -> Dict:
    """Benchmark one ring P2P step using NIXL READ (pull) transport."""
    try:
        from tensorrt_llm._torch.disaggregation.base.agent import (
            MemoryDescs,
            RegMemoryDescs,
            TransferRequest,
        )
        from tensorrt_llm._torch.disaggregation.nixl._agent_py import NixlTransferAgent
    except ImportError:
        return {"error": "NIXL not available (install nixl-cu13 or nixl-cu12)"}

    rank = dist.get_rank(group=group)
    global_rank = dist.get_rank()
    prev_ring_rank = (rank - 1) % world_size
    device_id = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_id}")

    # One contiguous buffer covers both ping-pong slots.
    kv_bufs = torch.zeros(2, slot_bytes // 2, dtype=torch.bfloat16, device=device)

    try:
        agent = NixlTransferAgent(name=f"bench_ring_{global_rank}", use_prog_thread=True)
    except Exception as e:
        return {"error": f"NixlTransferAgent init failed: {e}"}

    # Exchange agent descriptors.
    local_desc = agent.get_local_agent_desc()
    desc_t = torch.tensor(list(local_desc), dtype=torch.uint8, device=device)
    desc_len = torch.tensor([len(local_desc)], dtype=torch.int64, device=device)
    all_lens = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(all_lens, desc_len, group=group)
    max_len = int(max(int(tl.item()) for tl in all_lens))
    padded = torch.zeros(max_len, dtype=torch.uint8, device=device)
    padded[: len(local_desc)] = desc_t
    all_descs = [torch.zeros(max_len, dtype=torch.uint8, device=device) for _ in range(world_size)]
    dist.all_gather(all_descs, padded, group=group)
    prev_len = int(all_lens[prev_ring_rank].item())
    prev_desc = bytes(all_descs[prev_ring_rank][:prev_len].cpu().tolist())
    prev_global = dist.get_global_rank(group, prev_ring_rank)
    prev_name = f"bench_ring_{prev_global}"
    try:
        agent.load_remote_agent(prev_name, prev_desc)
    except Exception as e:
        return {"error": f"load_remote_agent failed: {e}"}

    # Register local kv_bufs.
    try:
        reg = RegMemoryDescs(
            type="VRAM", descs=[(kv_bufs.data_ptr(), kv_bufs.nbytes, device_id, "")]
        )
        agent.register_memory(reg)
    except Exception as e:
        return {"error": f"register_memory failed: {e}"}

    # All-gather base pointers.
    info = torch.tensor(
        [kv_bufs.data_ptr(), kv_bufs.nbytes, device_id], dtype=torch.int64, device=device
    )
    all_info = [torch.zeros(3, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(all_info, info, group=group)
    prev_base_ptr = int(all_info[prev_ring_rank][0].item())
    prev_dev = int(all_info[prev_ring_rank][2].item())
    slot_sz = kv_bufs[0].nbytes

    def fn():
        for cur in range(2):
            nxt = 1 - cur
            torch.cuda.current_stream().synchronize()
            dist.barrier(group=group)
            src = MemoryDescs("VRAM", [(prev_base_ptr + cur * slot_sz, slot_sz, prev_dev)])
            dst = MemoryDescs("VRAM", [(kv_bufs[nxt].data_ptr(), slot_sz, device_id)])
            req = TransferRequest(op="READ", src_descs=src, dst_descs=dst, remote_name=prev_name)
            status = agent.submit_transfer_requests(req)
            status.wait()
            torch.cuda.synchronize()

    try:
        times = _timed_iters(fn, warmup=warmup, iters=iters)
    except Exception as e:
        return {"error": f"NIXL transfer failed: {e}"}

    try:
        reg = RegMemoryDescs(
            type="VRAM", descs=[(kv_bufs.data_ptr(), kv_bufs.nbytes, device_id, "")]
        )
        agent.deregister_memory(reg)
    except Exception:
        pass

    return _stats(times)


def _ring_p2p_slot_bytes(shape_kv: tuple, dtype: torch.dtype) -> int:
    """Bytes for one ping-pong slot of kv_bufs: shape [2, B, S, H_kv, D]."""
    import math

    elems = math.prod(shape_kv)
    return elems * torch.finfo(dtype).bits // 8


def run_ring_p2p_benchmark(args, group: dist.ProcessGroup, world_size: int) -> Dict:
    """Benchmark ring P2P exchange (one step) across available transports."""
    rank = dist.get_rank()
    results = {}

    # Representative KV-slab sizes: [2, B, S/world_size, H_kv, D]
    # wan_1.3b: seq=7680, H_kv=5, D=128 (after CP split)
    # stdit3: seq=4096, H_kv=8, D=64
    configs = [
        ("wan_kv_slab", (2, 1, 7680 // world_size, 5, 128)),
        ("stdit3_kv_slab", (2, 1, 4096 // world_size, 8, 64)),
    ]
    backends = args.backend if args.backend != "all" else ["nccl", "ub", "nixl"]
    if isinstance(backends, str):
        backends = [backends]

    for name, shape in configs:
        slot_bytes = _ring_p2p_slot_bytes(shape, torch.bfloat16)
        if rank == 0:
            mb = slot_bytes / 1024**2
            print(f"\n  [{name}]  slot={mb:.1f} MB  shape={shape}")

        for transport in backends:
            label = f"{transport}_{name}"
            if transport == "nccl":
                stat = _bench_ring_p2p_nccl(group, world_size, slot_bytes, args.warmup, args.iters)
            elif transport == "nixl":
                stat = _bench_ring_p2p_nixl(group, world_size, slot_bytes, args.warmup, args.iters)
            elif transport == "ub":
                stat = {"error": "UB ring P2P micro-bench not yet wired (requires UB init context)"}
            else:
                stat = {"error": f"Unknown transport: {transport}"}
            results[label] = stat
            if rank == 0:
                if "error" in stat:
                    print(f"    {label:40s}  SKIP: {stat['error']}")
                else:
                    speedup = ""
                    nccl_key = f"nccl_{name}"
                    if (
                        transport != "nccl"
                        and nccl_key in results
                        and "error" not in results[nccl_key]
                    ):
                        ratio = results[nccl_key]["mean_ms"] / stat["mean_ms"]
                        speedup = f"  ({ratio:.2f}x vs NCCL)"
                    print(
                        f"    {label:40s}  mean={stat['mean_ms']:.3f} ms  "
                        f"median={stat['median_ms']:.3f} ms{speedup}"
                    )
    return results


def run_micro_benchmark(args) -> Dict:
    """Tier-1: no model weights — collective latency + ring P2P transport comparison."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    results = {}
    backend = getattr(args, "backend", "nccl")
    backends = backend if backend != "all" else ["nccl", "ub", "nixl"]
    if isinstance(backends, str):
        backends = [backends]

    # ---- Section A: Ulysses collective micro-benchmark ----
    collective_configs = [
        # (name, collective, shape=(B, S/world_size, H/world_size, D))
        ("flux_all_to_all", "all_to_all", (1, 4096 // world_size, 24 // world_size, 128)),
        ("flux_all_gather", "all_gather", (1, 4096 // world_size, 24, 128)),
        ("wan_all_to_all", "all_to_all", (1, 7680 // world_size, 40 // world_size, 128)),
        ("wan_all_gather", "all_gather", (1, 7680 // world_size, 40, 128)),
        ("all_reduce_small", "all_reduce", (1, 512, 512)),
    ]

    def _run_collective(label_prefix, bench_fn):
        for name, collective, shape in collective_configs:
            full_name = f"{label_prefix}_{name}"
            stat = bench_fn(
                group=dist.group.WORLD,
                world_size=world_size,
                shape=shape,
                dtype=torch.bfloat16,
                collective=collective,
                warmup=args.warmup,
                iters=args.iters,
            )
            results[full_name] = stat
            if rank == 0:
                if "error" in stat:
                    print(f"  {full_name:40s}  SKIP: {stat['error']}")
                else:
                    speedup = ""
                    nccl_key = f"nccl_{name}"
                    if label_prefix == "ub" and nccl_key in results:
                        ratio = results[nccl_key]["mean_ms"] / stat["mean_ms"]
                        speedup = f"  ({ratio:.2f}x vs NCCL)"
                    print(
                        f"  {full_name:40s}  mean={stat['mean_ms']:.3f} ms  "
                        f"median={stat['median_ms']:.3f} ms{speedup}"
                    )

    if "nccl" in backends:
        if rank == 0:
            print("\n--- Ulysses collectives: NCCL baseline ---")
        _run_collective("nccl", _bench_collective_one)

    if "ub" in backends:
        if rank == 0:
            print("\n--- Ulysses collectives: UserBuffers (UB) ---")
        _run_collective("ub", _bench_ub_alltoall_one)

    # ---- Section B: Ring P2P transport comparison ----
    if rank == 0:
        print("\n--- Ring attention P2P: NCCL vs UB vs NIXL ---")
    ring_args = argparse.Namespace(**vars(args))
    ring_args.backend = backends
    ring_results = run_ring_p2p_benchmark(ring_args, dist.group.WORLD, world_size)
    results.update({f"ring_{k}": v for k, v in ring_results.items()})

    return results


# ---------------------------------------------------------------------------
# Tier 2: Full pipeline benchmark
# ---------------------------------------------------------------------------


def _bench_pipeline_one(
    model_path: str,
    config_path: str,
    prompt: str,
    n_steps: int,
    warmup: int,
    iters: int,
    height: int,
    width: int,
) -> Dict:
    """Run trtllm VisualGen for n_steps, measure wall-clock per-image latency."""
    from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams

    args = VisualGenArgs.from_yaml(config_path)
    args.model = model_path

    vg = VisualGen(args)
    params = VisualGenParams(height=height, width=width, num_inference_steps=n_steps)

    # Warmup
    for _ in range(warmup):
        vg.generate(inputs=prompt, params=params)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        vg.generate(inputs=prompt, params=params)
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
            (
                "flux_baseline",
                args.flux_path,
                "examples/visual_gen/serve/configs/flux1.yml",
                1024,
                1024,
            )
        )
        benchmarks.append(
            (
                "flux_nccl_ub",
                args.flux_path,
                "examples/visual_gen/serve/configs/flux1_nccl_ub.yml",
                1024,
                1024,
            )
        )
    if args.wan_path:
        benchmarks.append(
            ("wan_baseline", args.wan_path, "examples/visual_gen/serve/configs/wan21.yml", 480, 832)
        )
        benchmarks.append(
            (
                "wan_nccl_ub",
                args.wan_path,
                "examples/visual_gen/serve/configs/wan21_nccl_ub.yml",
                480,
                832,
            )
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
    print(f"\n{'=' * 72}")
    print(f"Ring Transport Benchmark — {tier}")
    print(f"{'=' * 72}")
    print(f"{'Key':<38}  {'Mean ms':>10}  {'Median ms':>10}  {'Min ms':>8}")
    print(f"{'-' * 38}  {'-' * 10}  {'-' * 10}  {'-' * 8}")
    for k, v in results.items():
        if "error" in v:
            print(f"{k:<38}  SKIP: {v['error']}")
        else:
            print(f"{k:<38}  {v['mean_ms']:>10.3f}  {v['median_ms']:>10.3f}  {v['min_ms']:>8.3f}")
    print()

    if tier == "pipeline":
        for prefix in ("flux", "wan"):
            base = results.get(f"{prefix}_baseline")
            ub = results.get(f"{prefix}_nccl_ub")
            if base and ub and "error" not in base and "error" not in ub:
                speedup = base["mean_ms"] / ub["mean_ms"]
                delta_ms = base["mean_ms"] - ub["mean_ms"]
                print(f"{prefix.upper()} speedup: {speedup:.3f}x  ({delta_ms:+.1f} ms per image)")
    print(f"{'=' * 72}\n")


def _write_markdown_report(results: Dict, metadata: Dict, path: str) -> None:
    """Write a side-by-side comparison report in Markdown."""
    import datetime

    lines = []
    lines.append("# Ring Attention Transport Benchmark Report")
    lines.append("")
    lines.append(f"**Date:** {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**World size:** {metadata.get('world_size', '?')}")
    lines.append(f"**Tier:** {metadata.get('tier', '?')}")
    lines.append(
        f"**Warmup:** {metadata.get('warmup', '?')}  |  **Iters:** {metadata.get('iters', '?')}"
    )
    lines.append("")

    # Ring P2P section — separate by workload, compare transports side by side
    ring_keys = {k: v for k, v in results.items() if k.startswith("ring_")}
    if ring_keys:
        lines.append("## Ring P2P KV-Block Transfer (per ring step)")
        lines.append("")
        lines.append("Measures one full ring P2P exchange step latency across transports.")
        lines.append("Lower is better. Speedup is relative to NCCL.")
        lines.append("")

        workloads = {}
        for key, stat in ring_keys.items():
            # key format: ring_{transport}_{workload_name}
            parts = key[len("ring_") :].split("_", 1)
            if len(parts) == 2:
                transport, workload = parts
                workloads.setdefault(workload, {})[transport] = stat

        for workload, transport_stats in workloads.items():
            lines.append(f"### {workload}")
            lines.append("")
            lines.append("| Transport | Mean (ms) | Median (ms) | Min (ms) | vs NCCL |")
            lines.append("|-----------|----------:|------------:|---------:|--------:|")
            nccl_mean = transport_stats.get("nccl", {}).get("mean_ms")
            for transport in ("nccl", "ub", "nixl"):
                stat = transport_stats.get(transport)
                if stat is None:
                    continue
                if "error" in stat:
                    lines.append(f"| {transport.upper()} | — | — | — | {stat['error'][:40]} |")
                else:
                    vs = "—"
                    if nccl_mean and transport != "nccl":
                        ratio = nccl_mean / stat["mean_ms"]
                        vs = f"**{ratio:.2f}x**" if ratio > 1.0 else f"{ratio:.2f}x"
                    elif transport == "nccl":
                        vs = "baseline"
                    lines.append(
                        f"| {transport.upper()} | {stat['mean_ms']:.3f} | {stat['median_ms']:.3f} | "
                        f"{stat['min_ms']:.3f} | {vs} |"
                    )
            lines.append("")

    # Collectives section
    coll_keys = {k: v for k, v in results.items() if not k.startswith("ring_")}
    if coll_keys:
        lines.append("## Ulysses Collective Latency")
        lines.append("")
        lines.append("| Key | Transport | Mean (ms) | Median (ms) | vs NCCL |")
        lines.append("|-----|-----------|----------:|------------:|--------:|")
        nccl_coll = {}
        for k, v in coll_keys.items():
            if k.startswith("nccl_") and "error" not in v:
                nccl_coll[k[5:]] = v.get("mean_ms")
        for k, v in coll_keys.items():
            if "error" in v:
                continue
            parts = k.split("_", 1)
            transport, name = (parts[0], parts[1]) if len(parts) == 2 else ("?", k)
            vs = "baseline" if transport == "nccl" else "—"
            if transport != "nccl" and name in nccl_coll and nccl_coll[name]:
                ratio = nccl_coll[name] / v["mean_ms"]
                vs = f"**{ratio:.2f}x**" if ratio > 1.0 else f"{ratio:.2f}x"
            lines.append(
                f"| {name} | {transport.upper()} | {v['mean_ms']:.3f} | {v['median_ms']:.3f} | {vs} |"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- NIXL uses GDR (GPUDirect RDMA) for cross-node IB transfers; "
        "intra-node falls back to UCX shared memory."
    )
    lines.append("- UB (UserBuffers) uses NVLink one-sided push; requires NVLink topology.")
    lines.append("- NCCL uses `ncclSend`/`ncclRecv` via `torch.distributed.batch_isend_irecv`.")
    lines.append(
        "- The NIXL path includes a `dist.barrier` per step for correctness; "
        "this will be replaced with CUDA-event sync in follow-up work."
    )
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Markdown report written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backend",
        choices=["nccl", "ub", "nixl", "all"],
        default="all",
        help="Transport(s) to benchmark: nccl, ub (UserBuffers), nixl, or all (default).",
    )
    parser.add_argument(
        "--tier",
        choices=["micro", "pipeline"],
        default="micro",
        help="micro: collective + ring P2P microbenchmark (no weights); "
        "pipeline: full end-to-end (requires --flux-path or --wan-path)",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations")
    parser.add_argument("--steps", type=int, default=20, help="Denoising steps for pipeline tier")
    parser.add_argument(
        "--flux-path", type=str, default="", help="Path to FLUX.1-dev checkpoint dir"
    )
    parser.add_argument(
        "--wan-path", type=str, default="", help="Path to Wan2.1-T2V-1.3B-Diffusers checkpoint dir"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="bench_ring_transport_results.json",
        help="Output JSON path (rank-0 only)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="bench_ring_transport_report.md",
        help="Markdown report path (rank-0 only)",
    )
    parser.add_argument(
        "--nccl-cumem",
        action="store_true",
        default=False,
        help="Force NCCL_CUMEM_ENABLE=1 for this process",
    )
    args = parser.parse_args()

    if args.nccl_cumem:
        os.environ["NCCL_CUMEM_ENABLE"] = "1"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        ring_transport = os.environ.get("TRTLLM_RING_TRANSPORT", "auto")
        nccl_cumem = os.environ.get("NCCL_CUMEM_ENABLE", "0")
        print(f"\nRing Transport Benchmark  tier={args.tier}  backend={args.backend}")
        print(
            f"world_size={world_size}  TRTLLM_RING_TRANSPORT={ring_transport}  "
            f"NCCL_CUMEM_ENABLE={nccl_cumem}"
        )
        print(f"warmup={args.warmup}  iters={args.iters}\n")

    if args.tier == "micro":
        results = run_micro_benchmark(args)
    else:
        if not args.flux_path and not args.wan_path:
            parser.error("Pipeline tier requires at least one of --flux-path / --wan-path")
        results = run_pipeline_benchmark(args)

    if rank == 0:
        _print_summary(results, args.tier)
        metadata = {
            "tier": args.tier,
            "world_size": world_size,
            "backend": args.backend,
            "ring_transport_env": os.environ.get("TRTLLM_RING_TRANSPORT", "auto"),
            "nccl_cumem_enable": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
            "warmup": args.warmup,
            "iters": args.iters,
        }
        payload = {**metadata, "results": results}
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {args.out}")
        _write_markdown_report(results, metadata, args.report)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
