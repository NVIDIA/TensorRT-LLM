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
r"""Op-level benchmark for the one-model speculative-decoding sampling ops.

Measures the three call shapes the one-model path uses, for each
``advanced_sampling_mode``, over a sweep of vocabulary sizes, row counts and
sampling-filter combinations:

===============  ==================================================  ==========================
call shape       op                                                  call sites
===============  ==================================================  ==========================
``tokens``       ``sample_from_logits_op``                           target/draft samplers
``probs``        ``compute_probs_from_logits``                       target probs for rejection
``tokens_probs`` ``sampling_batch_spec_dec_one_model_for_rejection`` draft sampler + draft probs
===============  ==================================================  ==========================

Both eager and CUDA-graph-replay latency are reported: these ops run inside a
captured graph in production, so eager latency alone can mislead.

Register a new implementation by adding it to ``available_impls()``; nothing
else in this file needs to change.

Usage::

    # baseline sweep, saved for later comparison
    python tests/microbenchmarks/sampling_ops_perf.py --save baseline.json

    # candidate vs. the stored baseline, with the acceptance gates applied
    python tests/microbenchmarks/sampling_ops_perf.py --candidate fused \\
        --compare baseline.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as fi
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

# Disable sentinels, matching SpecMetadata._scan_one_model_sampling. A "neutral"
# row is NOT a greedy row: it samples, it just enables no filter (the
# temperature=1.0 / top_k unset / top_p unset case). Greedy rows never reach
# these ops -- the caller takes an argmax fast path instead.
DISABLE_TOPK = torch.iinfo(torch.int32).max
DISABLE_TOPP = 1.0
DISABLE_MINP = 0.0

CALL_SHAPES = ("tokens", "probs", "tokens_probs")


# ---------------------------------------------------------------------------
# Filter cases -- what a row asks the sampler to do, and how it is judged
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterCase:
    """One sampling-filter combination, plus the gate it is judged against.

    ``baseline_mode`` is the mode a deploy would actually have selected for this
    workload -- comparing an all-neutral batch against ``FULL`` would flatter the
    candidate, since a deploy with no filters would have configured
    ``NO_TOPK_NO_TOPP`` and paid for neither kernel.

    ``gate`` is the largest acceptable ``candidate / baseline`` latency ratio.
    ``None`` means there is nothing to compare against: the flashinfer chain
    cannot apply min_p at all, so any measurement of it for a min_p case is a
    lower bound on a strictly smaller feature set, not a baseline.
    """

    name: str
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    #: Apply the filters to half the rows only, leaving the rest neutral. This
    #: is the case the per-row skip exists for and that a per-deploy mode
    #: cannot express.
    mixed: bool = False
    baseline_mode: Optional[AdvancedSamplingMode] = None
    gate: Optional[float] = None
    gate_note: str = ""


FILTER_CASES: tuple[FilterCase, ...] = (
    FilterCase(
        name="neutral",
        baseline_mode=AdvancedSamplingMode.NO_TOPK_NO_TOPP,
        gate=1.15,
        gate_note="what a deploy pays for enabling the fused op while most requests filter nothing",
    ),
    FilterCase(
        name="top_k",
        top_k=50,
        baseline_mode=AdvancedSamplingMode.NO_TOPP,
        gate=1.25,
    ),
    FilterCase(
        name="top_p",
        top_p=0.9,
        baseline_mode=AdvancedSamplingMode.NO_TOPK,
        gate=1.25,
    ),
    FilterCase(
        name="top_k_top_p",
        top_k=50,
        top_p=0.9,
        baseline_mode=AdvancedSamplingMode.FULL,
        gate=1.25,
    ),
    FilterCase(
        name="min_p",
        min_p=0.05,
        gate_note="no baseline: the flashinfer chain cannot apply min_p",
    ),
    FilterCase(
        name="all_filters",
        top_k=50,
        top_p=0.9,
        min_p=0.05,
        gate_note="no baseline: the flashinfer chain cannot apply min_p",
    ),
    FilterCase(
        name="mixed_half_neutral",
        top_k=50,
        top_p=0.9,
        mixed=True,
        baseline_mode=AdvancedSamplingMode.FULL,
        gate=1.0,
        gate_note="must beat FULL, or the per-row skip has not earned its complexity",
    ),
)


@dataclass
class Inputs:
    logits: torch.Tensor
    temperatures: torch.Tensor
    top_ks: torch.Tensor
    top_ps: torch.Tensor
    min_ps: torch.Tensor
    seed: torch.Tensor
    offset: torch.Tensor


def make_inputs(case: FilterCase, rows: int, vocab: int, dtype: torch.dtype, device: str) -> Inputs:
    """Build one case's tensors in the layout the real call sites pass."""
    logits = torch.randn(rows, vocab, device=device, dtype=dtype) * 2.0
    temperatures = torch.full((rows,), 0.8, device=device, dtype=torch.float32)
    top_ks = torch.full((rows,), DISABLE_TOPK, device=device, dtype=torch.int32)
    top_ps = torch.full((rows,), DISABLE_TOPP, device=device, dtype=torch.float32)
    min_ps = torch.full((rows,), DISABLE_MINP, device=device, dtype=torch.float32)

    # A mixed batch filters the back half and leaves the front half neutral.
    active = slice(rows // 2, rows) if case.mixed else slice(0, rows)
    if case.top_k is not None:
        top_ks[active] = case.top_k
    if case.top_p is not None:
        top_ps[active] = case.top_p
    if case.min_p is not None:
        min_ps[active] = case.min_p

    # Both must be tensors: that is what makes the RNG state CUDA-graph legal.
    seed = torch.tensor([1234], dtype=torch.int64, device=device)
    offset = torch.tensor([0], dtype=torch.int64, device=device)
    return Inputs(logits, temperatures, top_ks, top_ps, min_ps, seed, offset)


# ---------------------------------------------------------------------------
# Implementations under test
# ---------------------------------------------------------------------------


@dataclass
class Impl:
    """One sampling backend, exposing the three call shapes.

    ``supports_min_p`` is not cosmetic: a measurement of a backend that silently
    drops min_p is not a baseline for a backend that applies it, and the report
    marks it so nobody reads the ratio as meaningful.
    """

    name: str
    call: Callable[[str, Inputs], Callable[[], Any]]
    supports_min_p: bool


def _flashinfer_impl(mode: AdvancedSamplingMode) -> Impl:
    """The current op chain, at one ``advanced_sampling_mode``.

    ``resolve_advanced_sampling_filters`` turning a disabled filter into ``None``
    is the whole point of the mode -- the op then omits that kernel rather than
    running a no-op one -- so it is applied here exactly as the call sites apply
    it, and not hoisted out of the timed region: it is a host-side tensor
    identity check, not work.
    """

    def call(shape: str, inp: Inputs) -> Callable[[], Any]:
        top_k, top_p = fi.resolve_advanced_sampling_filters(mode, inp.top_ks, inp.top_ps)
        if shape == "tokens":
            return lambda: fi.sample_from_logits_op(
                inp.logits, inp.temperatures, top_k, top_p, seed=inp.seed, offset=inp.offset
            )
        if shape == "probs":
            return lambda: fi.compute_probs_from_logits(inp.logits, inp.temperatures, top_k, top_p)
        if shape == "tokens_probs":
            return lambda: fi.sampling_batch_spec_dec_one_model_for_rejection(
                inp.logits, inp.temperatures, top_k, top_p, seed=inp.seed, offset=inp.offset
            )
        raise ValueError(f"unknown call shape: {shape}")

    return Impl(name=f"flashinfer:{mode.value}", call=call, supports_min_p=False)


def available_impls() -> dict[str, Impl]:
    """Return every available sampling backend."""
    impls: dict[str, Impl] = {}
    if IS_FLASHINFER_AVAILABLE:
        for mode in AdvancedSamplingMode:
            if mode.is_fused:
                continue
            impl = _flashinfer_impl(mode)
            impls[impl.name] = impl

    try:
        from tensorrt_llm._torch.pyexecutor.sampler.ops import fused
    except ImportError:
        return impls
    if not fused.is_available():
        return impls

    def call(shape: str, inp: Inputs) -> Callable[[], Any]:
        args = (inp.logits, inp.temperatures, inp.top_ks, inp.top_ps, inp.min_ps)
        if shape == "tokens":
            return lambda: fused.fused_sample_from_logits(*args, seed=inp.seed, offset=inp.offset)
        if shape == "probs":
            return lambda: fused.fused_compute_probs_from_logits(*args)
        if shape == "tokens_probs":
            return lambda: fused.fused_sample_from_logits_with_probs(
                *args, seed=inp.seed, offset=inp.offset
            )
        raise ValueError(f"unknown call shape: {shape}")

    impls["fused"] = Impl(name="fused", call=call, supports_min_p=True)
    return impls


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def time_eager(fn: Callable[[], Any], warmup: int, iters: int) -> float:
    """Median per-call latency in microseconds.

    Median rather than mean: the first iterations after a warmup can still catch
    an autotuning cache miss or a clock ramp, and one outlier should not move the
    number a gate is read off.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))


def time_graph(fn: Callable[[], Any], warmup: int, iters: int) -> tuple[Optional[float], str]:
    """Median replay latency in microseconds, or ``(None, reason)``.

    Capture is done on a side stream after a warmup, per the documented recipe:
    the ops are ``torch.compile``d and the first calls both compile and autotune,
    neither of which is capturable.
    """
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
    except RuntimeError as e:
        # Not a benchmark failure: a backend that cannot be captured is a real
        # finding about that backend, so it is reported rather than raised.
        return None, f"capture failed: {type(e).__name__}: {e}".split("\n")[0][:120]

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        graph.replay()
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)), ""


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


@dataclass
class Result:
    impl: str
    case: str
    shape: str
    rows: int
    vocab: int
    dtype: str
    eager_us: Optional[float]
    graph_us: Optional[float]
    note: str = ""
    #: False when the backend cannot apply every filter the case asks for, so
    #: its number describes less work than the case specifies.
    equivalent: bool = True

    def key(self) -> str:
        return f"{self.impl}|{self.case}|{self.shape}|{self.rows}|{self.vocab}|{self.dtype}"


def run_sweep(
    impls: Sequence[Impl],
    cases: Sequence[FilterCase],
    shapes: Sequence[str],
    rows_list: Sequence[int],
    vocab_list: Sequence[int],
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    skip_graph: bool,
    device: str,
) -> list[Result]:
    results: list[Result] = []
    total = len(impls) * len(cases) * len(shapes) * len(rows_list) * len(vocab_list)
    done = 0
    for vocab in vocab_list:
        for rows in rows_list:
            for case in cases:
                torch.manual_seed(0)
                inp = make_inputs(case, rows, vocab, dtype, device)
                for impl in impls:
                    for shape in shapes:
                        done += 1
                        # Carriage-return progress is for a terminal. A detached run
                        # redirects stderr to a file, where \r collapses the whole sweep
                        # onto one unreadable line -- and a detached run is how this is
                        # normally used, since it holds a GPU for minutes.
                        progress = (
                            f"  [{done}/{total}] {impl.name} {case.name} {shape} "
                            f"rows={rows} vocab={vocab}"
                        )
                        if sys.stderr.isatty():
                            print(progress, end="\r", file=sys.stderr, flush=True)
                        elif done % 25 == 0 or done == total:
                            print(progress, file=sys.stderr, flush=True)
                        equivalent = impl.supports_min_p or case.min_p is None
                        fn = impl.call(shape, inp)
                        eager = time_eager(fn, warmup, iters)
                        graph, note = (None, "graph timing skipped")
                        if not skip_graph:
                            graph, note = time_graph(fn, warmup, iters)
                        if not equivalent:
                            note = ("; ".join(x for x in (note, "min_p NOT applied") if x)).strip()
                        results.append(
                            Result(
                                impl=impl.name,
                                case=case.name,
                                shape=shape,
                                rows=rows,
                                vocab=vocab,
                                dtype=str(dtype).replace("torch.", ""),
                                eager_us=eager,
                                graph_us=graph,
                                note=note,
                                equivalent=equivalent,
                            )
                        )
                        del fn
                        torch.cuda.empty_cache()
    if sys.stderr.isatty():
        print(" " * 100, end="\r", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float]) -> str:
    return "     n/a" if value is None else f"{value:8.1f}"


def print_table(results: Sequence[Result]) -> None:
    """One block per (vocab, rows); rows are (case, shape, impl)."""
    groups: dict[tuple[int, int], list[Result]] = {}
    for r in results:
        groups.setdefault((r.vocab, r.rows), []).append(r)

    for (vocab, rows), items in sorted(groups.items()):
        print(f"\nvocab={vocab}  rows={rows}  dtype={items[0].dtype}")
        print(f"  {'case':<20} {'shape':<13} {'impl':<28} {'eager_us':>8} {'graph_us':>9}  note")
        print("  " + "-" * 98)
        for r in items:
            flag = "" if r.equivalent else "*"
            print(
                f"  {r.case:<20} {r.shape:<13} {r.impl + flag:<28} "
                f"{_fmt(r.eager_us)} {_fmt(r.graph_us)}  {r.note}"
            )
    if any(not r.equivalent for r in results):
        print("\n  * this backend does not apply min_p; its number is for a smaller feature set")


def evaluate_gates(
    results: Sequence[Result], candidate: str, metric: str
) -> tuple[list[str], list[str]]:
    """Apply each case's gate to ``candidate`` against its own baseline mode.

    Returns ``(passes, failures)`` as human-readable lines. A case whose gate is
    ``None`` is reported as unjudged rather than silently passed -- min_p has no
    baseline, and pretending otherwise is how a regression hides.
    """
    by_key: dict[tuple[str, str, str, int, int], Result] = {
        (r.impl, r.case, r.shape, r.rows, r.vocab): r for r in results
    }
    cases_by_name = {c.name: c for c in FILTER_CASES}
    passes: list[str] = []
    failures: list[str] = []

    for r in results:
        if r.impl != candidate:
            continue
        case = cases_by_name[r.case]
        label = f"{r.case:<20} {r.shape:<13} rows={r.rows:<5} vocab={r.vocab}"
        if case.gate is None or case.baseline_mode is None:
            passes.append(f"  [unjudged] {label}  ({case.gate_note})")
            continue
        base = by_key.get(
            (f"flashinfer:{case.baseline_mode.value}", r.case, r.shape, r.rows, r.vocab)
        )
        cand_us = getattr(r, metric)
        base_us = getattr(base, metric) if base is not None else None
        if base_us is None or cand_us is None:
            passes.append(f"  [unjudged] {label}  (no {metric} for the baseline)")
            continue
        ratio = cand_us / base_us
        line = (
            f"  {label}  {ratio:5.2f}x vs {case.baseline_mode.value} "
            f"(gate {case.gate:.2f}x, {cand_us:.1f}us vs {base_us:.1f}us)"
        )
        (passes if ratio <= case.gate else failures).append(line)
    return passes, failures


def save(results: Sequence[Result], path: str) -> None:
    with open(path, "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)
    print(f"\nsaved {len(results)} measurements to {path}")


def load(path: str) -> list[Result]:
    with open(path) as f:
        return [Result(**row) for row in json.load(f)]


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--vocab",
        type=int,
        nargs="+",
        default=[32000, 131072],
        help="vocabulary sizes (default: 32000 131072; --full adds 152064)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32, 64],
        help=(
            "rows of the logits tensor, i.e. independent distributions per call. This is "
            "NOT the serving batch size: the draft sampler gets one row per request "
            "(rows == batch), while the target gets batch x (draft_len + 1). "
            "The default is the small-batch regime, which is where this kernel is weakest "
            "-- one block owns one row, so below the SM count the wall time stops falling "
            "and most of the GPU sits idle. Use --full for the regression sweep that also "
            "covers the large batches, where the kernel already wins."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="the complete sweep from the design: vocab 32k/128k/152k, rows 1..2048. Slow.",
    )
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--shapes", nargs="+", default=list(CALL_SHAPES), choices=CALL_SHAPES)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[c.name for c in FILTER_CASES],
        choices=[c.name for c in FILTER_CASES],
    )
    parser.add_argument(
        "--impls",
        nargs="+",
        default=None,
        help="default: every available backend. Names as printed by --list-impls.",
    )
    parser.add_argument("--list-impls", action="store_true")
    parser.add_argument(
        "--candidate",
        default=None,
        help="backend to judge against each case's baseline mode (e.g. 'fused')",
    )
    parser.add_argument(
        "--gate-metric",
        default="graph_us",
        choices=["graph_us", "eager_us"],
        help="which measurement the gates read (default graph_us: production replays a graph)",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--skip-graph", action="store_true", help="eager timing only")
    parser.add_argument("--save", default=None, help="write results as JSON")
    parser.add_argument("--compare", default=None, help="merge a saved JSON in before gating")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    impls_by_name = available_impls()
    if args.list_impls:
        for name, impl in impls_by_name.items():
            print(f"{name:<28} min_p={'yes' if impl.supports_min_p else 'no'}")
        return 0
    if not impls_by_name:
        print("no sampling backend available (flashinfer missing?)", file=sys.stderr)
        return 1

    selected = args.impls or list(impls_by_name)
    unknown = [n for n in selected if n not in impls_by_name]
    if unknown:
        print(f"unknown impls: {unknown}; available: {list(impls_by_name)}", file=sys.stderr)
        return 1
    impls = [impls_by_name[n] for n in selected]

    vocab_list = [32000, 131072, 152064] if args.full else args.vocab
    rows_list = [1, 4, 8, 16, 32, 40, 64, 128, 256, 1024, 2048] if args.full else args.rows
    cases = [c for c in FILTER_CASES if c.name in args.cases]
    dtype = getattr(torch, args.dtype)

    if IS_FLASHINFER_AVAILABLE:
        # Build flashinfer's kernels now; the first call otherwise runs nvcc
        # inline and would land inside a timed region.
        fi.warmup_sampling_module()

    print(
        f"device={torch.cuda.get_device_name()}  dtype={args.dtype}  "
        f"warmup={args.warmup} iters={args.iters}"
    )
    print(f"impls={[i.name for i in impls]}")

    results = run_sweep(
        impls,
        cases,
        args.shapes,
        rows_list,
        vocab_list,
        dtype,
        args.warmup,
        args.iters,
        args.skip_graph,
        "cuda",
    )
    print_table(results)

    if args.save:
        save(results, args.save)

    if args.candidate:
        merged = list(results)
        if args.compare:
            have = {r.key() for r in results}
            merged += [r for r in load(args.compare) if r.key() not in have]
        passes, failures = evaluate_gates(merged, args.candidate, args.gate_metric)
        print(f"\ngates for '{args.candidate}' on {args.gate_metric}")
        print("-" * 100)
        for line in passes:
            print(line)
        for line in failures:
            print(f"  FAIL {line.strip()}")
        print(f"\n{len(failures)} failing, {len(passes)} passing/unjudged")
        return 1 if failures else 0

    print("\nno --candidate given: this is a baseline run, nothing to gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
