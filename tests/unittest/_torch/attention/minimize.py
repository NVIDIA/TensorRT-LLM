# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Delta-debugging minimizer for captured attention cases.

A real-workload failure usually reproduces only at the last failing kernel call.
Capture-all records every step; this tool takes a failing case and greedily
shrinks it (batch, seq lens, cached tokens, heads, page size) while the
backend-vs-Vanilla-golden failure persists, yielding a minimal reproducer to
commit as a fixture. Minimization also deduplicates — minimal forms converge.

Reproduces *structural* bugs (shape/config/masking/paging/illegal-access). It
will not reproduce value-dependent numerical bugs, since replay uses random data.

CLI:
    python -m minimize <cases.jsonl> [--backend TRTLLM] [--index -1]
Prints the minimized case as JSON to stdout.
"""

import dataclasses as dc
import json
from typing import Iterator

import torch
from attention_test_harness import ATOL, FP8_ATOL, RTOL, BackendCase, generate_inputs, run_backend
from capability_matrix import unsupported_reason


def case_fails(case: BackendCase, backend: str, *, seed: int = 0) -> bool:
    """True if ``backend`` diverges from the Vanilla golden (or crashes) on ``case``."""
    kv_dtype = case.compute_dtype if case.cache == "none" else case.kv_torch_dtype
    try:
        inputs = generate_inputs(case, seed)
        golden = run_backend(case, "VANILLA", inputs, kv_dtype=case.compute_dtype)
        out = run_backend(case, backend, inputs, kv_dtype=kv_dtype)
    except Exception:
        # A crash is a failure worth preserving/minimizing.
        return True
    atol = FP8_ATOL if kv_dtype == torch.float8_e4m3fn else ATOL
    try:
        torch.testing.assert_close(out, golden, atol=atol, rtol=RTOL)
        return False
    except AssertionError:
        return True


def _shrink_candidates(case: BackendCase) -> Iterator[BackendCase]:
    """Yield strictly-smaller variants, biggest-impact reductions first."""
    n = case.num_seqs

    # 1. Drop a sequence (reduce batch).
    if n > 1:
        for drop in range(n):
            keep = [i for i in range(n) if i != drop]
            new_contexts = sum(1 for i in keep if i < case.num_contexts)
            yield dc.replace(
                case,
                seq_lens=[case.seq_lens[i] for i in keep],
                num_cached_tokens=[case.num_cached_tokens[i] for i in keep],
                num_contexts=new_contexts,
            )

    # 2. Halve each query length.
    for i in range(n):
        if case.seq_lens[i] > 1:
            sl = list(case.seq_lens)
            sl[i] = max(1, sl[i] // 2)
            yield dc.replace(case, seq_lens=sl)

    # 3. Halve each cached length.
    for i in range(n):
        if case.num_cached_tokens[i] > 0:
            nc = list(case.num_cached_tokens)
            nc[i] = nc[i] // 2
            yield dc.replace(case, num_cached_tokens=nc)

    # 4. Reduce head counts (preserve the GQA ratio when possible).
    if case.num_kv_heads > 1 and case.num_heads % case.num_kv_heads == 0:
        ratio = case.num_heads // case.num_kv_heads
        nkv = case.num_kv_heads // 2
        yield dc.replace(case, num_kv_heads=nkv, num_heads=nkv * ratio)
    elif case.num_heads > 1 and case.num_kv_heads == 1:
        yield dc.replace(case, num_heads=max(1, case.num_heads // 2))

    # 5. Shrink page size (more/fewer blocks per sequence).
    if case.cache != "none" and case.page_size > 16:
        yield dc.replace(case, page_size=case.page_size // 2)


def minimize_case(
    case: BackendCase, backend: str, *, seed: int = 0, max_rounds: int = 100
) -> BackendCase:
    """Greedily reduce ``case`` while ``backend`` keeps failing against golden."""
    if not case_fails(case, backend, seed=seed):
        raise ValueError(f"Case does not fail for backend {backend}; nothing to minimize.")
    cur = case
    for _ in range(max_rounds):
        progressed = False
        for cand in _shrink_candidates(cur):
            if unsupported_reason(backend, cand) is not None:
                continue
            if case_fails(cand, backend, seed=seed):
                cur = cand
                progressed = True
                break
        if not progressed:
            break
    return cur


def _main() -> None:
    import argparse

    from case_io import iter_case_specs

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="captured .jsonl / .json or a fixtures dir")
    ap.add_argument("--backend", default="TRTLLM")
    ap.add_argument(
        "--index", type=int, default=-1, help="which captured case to minimize (default: last)"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cases = list(iter_case_specs(args.path))
    if not cases:
        raise SystemExit(f"No cases found in {args.path}")
    case = cases[args.index]
    mini = minimize_case(case, args.backend, seed=args.seed)
    print(json.dumps(mini.to_dict(), indent=2))


if __name__ == "__main__":
    _main()
