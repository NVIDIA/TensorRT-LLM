# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep the MiniMax-M3 gate projection from decode to full prefill.

    python3 tests/microbenchmarks/minimax_m3_gate_gemm/bench.py

The gate is [num_tokens, 6144] bf16 by [128, 6144] fp32, once per sparse layer.
At 32 tokens it is latency-bound and the activation fits in L2; at 16384 it is a
200MB stream where only reading that stream once matters. One sweep over both
ends shows where each candidate stops being the right answer.

Add --cute to include the CuTe DSL kernel, --only to filter candidates by
substring, and --tokens to change the sweep.

Where the crossovers land, on a B200
------------------------------------

From 8 tokens up the CuTe kernel is ahead of the TF32 path the model runs today,
by 1.14x at the bottom and 7.6x at 16384, and it is 25x to 70x closer to the FP64
reference the whole way. At 1 token it is still behind, so the 1 to 16 token band
stays with the Triton GEMV, which answers a 1-token router in 1.97us against
7.07us here.

Two settings carry that result. Splitting K is what lifts the small-M end, from
8.9us to 2.8us at 32 tokens, and the default stops at 4 partitions because
reducing the partials costs about 1.8us. The tile shape switches at 8192 tokens,
above which a tile spanning all of N lets the epilogue fold the weight terms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):  # allow running the file directly
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    __package__ = "tests.microbenchmarks.minimax_m3_gate_gemm"

from .baselines import (  # noqa: E402
    Candidate,
    reference_candidates,
    torch_candidates,
    triton_gemv_candidate,
)
from .harness import (  # noqa: E402
    M3_EXPERTS,
    M3_HIDDEN,
    M3_SPARSE_LAYERS,
    GateProblem,
    evaluate,
    format_table,
    make_inputs,
    measure_achievable_bandwidth_gbs,
    measure_replay_floor,
    reference,
)

DEFAULT_TOKENS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def _collect(args) -> list[Candidate]:
    candidates: list[Candidate] = list(torch_candidates())

    triton = triton_gemv_candidate()
    if triton is not None:
        candidates.append(triton)

    if args.cute:
        from .cute_candidates import cute_candidates

        candidates.extend(cute_candidates())

    if args.refs:
        candidates.extend(reference_candidates())

    if args.only:
        candidates = [c for c in candidates if any(s in c.name for s in args.only)]
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=DEFAULT_TOKENS)
    parser.add_argument("--hidden-size", type=int, default=M3_HIDDEN)
    parser.add_argument("--num-experts", type=int, default=M3_EXPERTS)
    parser.add_argument("--layers", type=int, default=M3_SPARSE_LAYERS)
    # The gate's GEMM runs on the TF32 tensor cores in production, so that is
    # what a replacement has to beat, on accuracy as much as on time.
    parser.add_argument("--baseline", default="cast + cublas tf32")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cute", action="store_true", help="Include the CuTe DSL kernel.")
    parser.add_argument(
        "--refs",
        action="store_true",
        help="Include unachievable reference points that bound the design space.",
    )
    parser.add_argument("--only", nargs="+", help="Substring filter over candidate names.")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Sweep CuTe DSL tile/cluster shapes at each --tokens value instead of comparing.",
    )
    parser.add_argument("--terms", type=int, default=2, help="Weight terms to tune for.")
    args = parser.parse_args()

    torch.cuda.init()

    if args.tune:
        _run_tune(args)
        return

    candidates = _collect(args)

    floor = measure_replay_floor()
    bandwidth = measure_achievable_bandwidth_gbs()
    print(f"device      : {torch.cuda.get_device_name(0)}")
    print(
        f"problem     : [M, {args.hidden_size}] bf16 x [{args.num_experts}, {args.hidden_size}] fp32"
    )
    print(
        f"harness     : replay floor {floor:.2f} us/call, copy bandwidth {bandwidth / 1000:.2f} TB/s"
    )
    print(f"baseline    : {args.baseline}")

    summary: dict[str, dict[int, float]] = {c.name: {} for c in candidates}
    for num_tokens in args.tokens:
        problem = GateProblem(num_tokens, args.hidden_size, args.num_experts)
        x, w = make_inputs(problem)
        ref = reference(x, w)

        results = []
        for cand in candidates:
            if cand.max_tokens is not None and num_tokens > cand.max_tokens:
                continue
            with cand.context():
                results.append(
                    evaluate(cand.name, cand.build(x, w), ref, warmup=args.warmup, iters=args.iters)
                )
            summary[cand.name][num_tokens] = results[-1].micros

        print()
        print(format_table(problem, results, baseline=args.baseline, bandwidth_gbs=bandwidth))
        del x, w, ref
        torch.cuda.empty_cache()

    _print_summary(summary, args)


def _run_tune(args) -> None:
    """Best tile and cluster shape per token count, for `default_tactic`."""
    from .cute_candidates import tune
    from .harness import time_us

    print(f"tuning the CuTe DSL gate GEMM at {args.terms} weight terms\n")
    print(
        f"{'tokens':>7s}  {'best us':>8s}  {'tactic':<28s} {'split_k':>7s}  "
        f"{'epilogue':<8s}  runners-up"
    )
    for num_tokens in args.tokens:
        problem = GateProblem(num_tokens, args.hidden_size, args.num_experts)
        x, w = make_inputs(problem)
        ranked = tune(x, w, args.terms, time_us)
        if not ranked:
            print(f"{num_tokens:7d}  {'-':>8s}  no valid tactic")
            continue
        best_us, best = ranked[0]
        runners = ", ".join(f"{us:.1f}us {t}" for us, t in ranked[1:3])
        folds = best[1][1] == args.terms * args.num_experts and best[3] == 1
        print(
            f"{num_tokens:7d}  {best_us:8.2f}  {str(best[:3]):<28s} "
            f"{best[3]:>7d}  {'fold' if folds else 'unfused':<8s}  {runners}"
        )
        del x, w
        torch.cuda.empty_cache()


def _print_summary(summary: dict[str, dict[int, float]], args) -> None:
    base = summary.get(args.baseline, {})
    if not base:
        return
    print(f"\n\nus/step saved against '{args.baseline}', over {args.layers} sparse layers")
    header = f"{'candidate':<28s}" + "".join(f"{t:>9d}" for t in args.tokens)
    print(header)
    for name, times in summary.items():
        if name == args.baseline:
            continue
        cells = []
        for t in args.tokens:
            if t not in times or t not in base:
                cells.append(f"{'-':>9s}")
            else:
                cells.append(f"{(base[t] - times[t]) * args.layers:9.0f}")
        print(f"{name:<28s}" + "".join(cells))


if __name__ == "__main__":
    main()
