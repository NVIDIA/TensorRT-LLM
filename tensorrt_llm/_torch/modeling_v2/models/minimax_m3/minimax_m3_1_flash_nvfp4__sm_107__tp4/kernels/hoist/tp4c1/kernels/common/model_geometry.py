# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared constants, geometry and timing for the G7 MiniMax-M3 decoder-block harness."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import torch
import torch.distributed as dist

H = 6144
D = 128
E = 128
TOP_K = 4
ROUTE_SCALE = 2.0
EPS = 1e-6
SWIGLU_ALPHA, SWIGLU_LIMIT, SWIGLU_BETA = 1.702, 7.0, 1.0
QLEN = 4  # EAGLE3 k=3 verify rows per request


@dataclass(frozen=True)
class Geo:
    """Per-rank geometry of one TP world."""

    tp: int

    @property
    def nh(self) -> int:  # KV heads == index heads per rank
        return 4 // self.tp

    @property
    def hq(self) -> int:
        return 16 * self.nh

    @property
    def qkv_n(self) -> int:  # [q | k | v | index_q | index_k]
        return (self.hq + 2 * self.nh + self.nh + 1) * D

    @property
    def o_k(self) -> int:
        return self.hq * D

    @property
    def inter(self) -> int:  # routed and shared intermediate per rank
        return 3072 // self.tp


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def log0(*a):
    if rank() == 0:
        print(*a, flush=True)


class L2Flush:
    """Evicts L2 by writing a buffer larger than it (SM107 L2 is 126 MiB)."""

    def __init__(self, mib: int = 512):
        self.buf = torch.empty(mib << 20, dtype=torch.uint8, device="cuda")

    def __call__(self):
        self.buf.fill_(1)


def capture(fn, warm: int = 2):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    return g


PREROLL_CYCLES = 200000  # ~85 us at 2.36 GHz


def time_graph(g, reps: int, flush: L2Flush | None, barrier: bool) -> dict:
    """Per-replay device time (us) via CUDA events; ``flush`` runs before the start event."""
    ts = []
    for i in range(reps + 3):
        if barrier and dist.is_initialized():
            dist.barrier()
        if flush is not None:
            flush()
        torch.cuda._sleep(
            PREROLL_CYCLES
        )  # GPU busy while the host submits event + replay: host latency stays outside
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        if i >= 3:
            ts.append(a.elapsed_time(b) * 1e3)
    ts.sort()
    q = lambda p: ts[min(len(ts) - 1, int(p * len(ts)))]  # noqa: E731
    return {"median": statistics.median(ts), "p10": q(0.10), "p90": q(0.90), "n": len(ts)}


def max_over_ranks(x: float) -> float:
    if not dist.is_initialized():
        return x
    t = torch.tensor([x], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))
