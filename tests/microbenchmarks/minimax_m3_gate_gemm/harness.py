# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Timing and accuracy utilities for the gate-projection microbenchmarks.

Measurement here spans three orders of magnitude in M. At 32 tokens a candidate
runs in a couple of microseconds, the same order as the 6us host cost of one
`graph.replay()`. At 16384 it runs for over a hundred, and a graph of 200 calls
would occupy the GPU for 20ms per replay. So timing always happens inside a CUDA
graph, which keeps launch cost out of the number, with the calls per graph chosen
from a target duration rather than a constant.

Results come back as `Measurement`, which carries accuracy alongside runtime.
Several candidates buy speed with mantissa bits, and microseconds alone would
make them look better than they are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

# MiniMax-M3: 128 experts over a 6144-wide hidden state, on each of 57 sparse
# layers. The gate runs once per layer per forward step.
M3_HIDDEN = 6144
M3_EXPERTS = 128
M3_SPARSE_LAYERS = 57


@dataclass(frozen=True)
class GateProblem:
    """One gate projection: [num_tokens, hidden] bf16 by [experts, hidden] fp32."""

    num_tokens: int
    hidden_size: int = M3_HIDDEN
    num_experts: int = M3_EXPERTS

    def traffic_bytes(self) -> int:
        """Compulsory global traffic for a kernel that reads the bf16 activation once.

        The 3MB weight is shared by every CTA, so past the first tile it is an L2
        hit rather than HBM traffic. Counting it once is the optimistic reading,
        and it is negligible against the activation.
        """
        activation = self.num_tokens * self.hidden_size * 2
        weight = self.num_experts * self.hidden_size * 4
        out = self.num_tokens * self.num_experts * 4
        return activation + weight + out

    def flops(self) -> int:
        return 2 * self.num_tokens * self.hidden_size * self.num_experts


@dataclass
class Measurement:
    name: str
    micros: float
    max_rel_err: float
    rms_rel_err: float
    failed: str | None = None


def _graph_calls_for(fn: Callable[[], object], target_ms: float) -> int:
    """Pick calls per graph so one replay lasts roughly `target_ms`."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(10):
        fn()
    end.record()
    torch.cuda.synchronize()
    per_call_ms = start.elapsed_time(end) / 10.0
    if per_call_ms <= 0:
        return 200
    return max(1, min(200, int(target_ms / per_call_ms)))


def time_us(
    fn: Callable[[], object],
    *,
    warmup: int = 10,
    iters: int = 50,
    target_graph_ms: float = 2.0,
) -> float:
    """Microseconds per call, measured over a CUDA graph of repeated calls."""
    calls = _graph_calls_for(fn, target_graph_ms)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / (iters * calls)


def measure_replay_floor(iters: int = 50) -> float:
    """What the harness reports for a kernel that does nothing.

    A candidate landing near this number is not being resolved by the harness.
    """
    scratch = torch.zeros(1, device="cuda")
    return time_us(lambda: scratch.add_(1.0), iters=iters)


def measure_achievable_bandwidth_gbs() -> float:
    """Streaming copy bandwidth, as the denominator for the roofline column.

    Measured rather than taken from the datasheet, so the percentage of HBM
    column compares against what this machine delivers.
    """
    n = 256 * 1024 * 1024 // 4  # 256MB in, 256MB out
    src = torch.empty(n, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    micros = time_us(lambda: dst.copy_(src), iters=20, target_graph_ms=20.0)
    return (2 * n * 4) / (micros * 1e-6) / 1e9


def make_inputs(problem: GateProblem, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Activation and router weight, scaled like the real ones.

    Hidden states out of an RMSNorm are roughly unit-variance, and the router
    weight is a small `nn.Linear` init. The scale matters for the accuracy
    column: it sets how much cancellation the 6144-long dot product sees.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        problem.num_tokens,
        problem.hidden_size,
        device="cuda",
        dtype=torch.float32,
        generator=gen,
    ).to(torch.bfloat16)
    w = torch.randn(
        problem.num_experts,
        problem.hidden_size,
        device="cuda",
        dtype=torch.float32,
        generator=gen,
    ) * (1.0 / math.sqrt(problem.hidden_size))
    return x, w


def reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """FP64 ground truth for the accuracy columns.

    The bf16 activation is exact in fp64, so this isolates what each candidate
    does to the weight and to the accumulation order, which is the whole numerics
    question for this GEMM.
    """
    return (x.double() @ w.double().t()).float()


def rel_errors(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """Max and RMS error, relative to the RMS magnitude of the reference.

    Router logits straddle zero, so normalising each element by itself would
    report enormous errors on the ones near zero, whose exact value the top-k
    does not care about. Normalising by the RMS of the whole tensor answers the
    question that matters: how big is the error next to a typical logit.
    """
    ref64 = ref.double()
    err = (got.double() - ref64).abs()
    scale = ref64.pow(2).mean().sqrt().item()
    if scale == 0:
        return 0.0, 0.0
    return err.max().item() / scale, err.pow(2).mean().sqrt().item() / scale


def evaluate(
    name: str,
    fn: Callable[[], torch.Tensor],
    ref: torch.Tensor,
    *,
    warmup: int = 10,
    iters: int = 50,
) -> Measurement:
    """Check a candidate against `ref`, then time it."""
    try:
        out = fn()
    except Exception as exc:  # noqa: BLE001 - a broken candidate should not stop the sweep
        return Measurement(
            name, math.inf, math.nan, math.nan, failed=f"{type(exc).__name__}: {exc}"
        )

    if out.shape != ref.shape:
        return Measurement(
            name,
            math.inf,
            math.nan,
            math.nan,
            failed=f"shape {tuple(out.shape)} != {tuple(ref.shape)}",
        )

    max_err, rms_err = rel_errors(out, ref)
    try:
        micros = time_us(fn, warmup=warmup, iters=iters)
    except Exception as exc:  # noqa: BLE001
        return Measurement(name, math.inf, max_err, rms_err, failed=f"{type(exc).__name__}: {exc}")
    return Measurement(name, micros, max_err, rms_err)


def format_table(
    problem: GateProblem,
    results: Sequence[Measurement],
    *,
    baseline: str,
    bandwidth_gbs: float,
) -> str:
    """One block per token count: time, speedup, roofline occupancy, accuracy."""
    base = next((r for r in results if r.name == baseline and r.failed is None), None)
    floor_us = problem.traffic_bytes() / (bandwidth_gbs * 1e9) * 1e6

    lines = [
        f"M = {problem.num_tokens}  "
        f"({problem.traffic_bytes() / 2**20:.1f} MiB minimum traffic, "
        f"HBM-bound floor {floor_us:.1f} us)",
        f"  {'candidate':<28s} {'us':>9s} {'vs base':>8s} {'% of HBM':>9s} "
        f"{'max rel':>10s} {'rms rel':>10s}",
    ]
    for r in results:
        if r.failed is not None:
            lines.append(f"  {r.name:<28s} {'FAILED':>9s}   {r.failed}")
            continue
        speedup = f"{base.micros / r.micros:.2f}x" if base else "-"
        pct_hbm = f"{100.0 * floor_us / r.micros:.0f}%"
        lines.append(
            f"  {r.name:<28s} {r.micros:9.2f} {speedup:>8s} {pct_hbm:>9s} "
            f"{r.max_rel_err:10.2e} {r.rms_rel_err:10.2e}"
        )
    return "\n".join(lines)
