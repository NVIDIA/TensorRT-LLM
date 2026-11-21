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
r"""Benchmark of one-model rejection sampling: the reference path against the fused kernels.

The draft step and the verify step of one-model rejection sampling have two implementations:

- the reference path: flashinfer softmax and sampling, a scatter into and a gather from
  ``draft_probs``, then ``chain_speculative_sampling``;
- the fused Triton kernels of ``fast_rejection_sampling``.

`SpecWorkerBase` takes the fused kernels wherever they cover the step. This script calls its two
entry points on the same inputs, once held on the reference path and once on the fused path:

- draft: ``sample_draft_tokens`` on block draft logits ``[B, K, V]``, as DFlash and DSpark
  without a Markov head call it. It draws the draft tokens and stores their distributions in
  ``draft_probs``. A DSpark Markov head draws its drafts through ``markov_sampled_chain``
  instead, so its draft step is not measured here.
- verify: ``_accept_draft_tokens`` on target logits ``[B * (K + 1), V]``, for every drafter.

The defaults are Kimi K3 shapes: a vocabulary of 163,840 tokens and fp32 logits, in the
temperature-only mode the fused kernels cover (``no_topk_no_topp``). The sweep
covers 1 to 512 generation requests per GPU, 1 to 16 draft tokens per request, temperatures 0.5,
1 and 1.5, and four logit shapes.

Each step is timed as a CUDA-graph replay, as it runs in production, and eagerly. The logits
are synthetic. Each shape puts a row's tokens in a random order and gives them Zipf-shaped
logits ``-alpha * ln(rank)``, or Gaussian logits for ``flat``; the output prints how many tokens
hold 95% of the probability for each shape.

Usage::

    python tests/microbenchmarks/spec_fast_rejection_sampling_perf.py --save fast_rs.json
    python tests/microbenchmarks/spec_fast_rejection_sampling_perf.py \\
        --batch-sizes 1 16 --draft-lens 3 7 --temperatures 1.0 --dists flat
    python tests/microbenchmarks/spec_fast_rejection_sampling_perf.py --load a.json b.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from tensorrt_llm._torch.speculative import interface as itf
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode
from tensorrt_llm.logger import logger

KIMI_K3_VOCAB = 163840
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
DEFAULT_DRAFT_LENS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16]
DEFAULT_TEMPERATURES = [0.5, 1.0, 1.5]
MODES = {"no_topk_no_topp": AdvancedSamplingMode.NO_TOPK_NO_TOPP}
# Zipf exponent of each logit shape; None is Gaussian logits.
DISTS = {"peaked": 2.0, "medium": 1.5, "longtail": 1.1, "flat": None}
IMPLS = ("ref", "fused")


class _Worker(itf.SpecWorkerBase):
    """A `SpecWorkerBase` without a model, to call its draft and verify entry points."""

    def __init__(self, draft_len: int):
        super().__init__()
        self.mapping = None
        self._draft_len = draft_len

    @property
    def max_draft_len(self):
        return self._draft_len

    def _forward_impl(self, *args, **kwargs):
        raise NotImplementedError


def make_metadata(
    num_gens: int, draft_len: int, vocab: int, mode: AdvancedSamplingMode, temperature: float
) -> itf.SpecMetadata:
    """Spec metadata of ``num_gens`` generation requests with rejection-sampling buffers.

    Every request has temperature ``temperature``, no top-k or top-p, and its own slot, seed and
    offset.
    """
    md = itf.SpecMetadata(
        max_num_requests=num_gens, max_draft_len=draft_len, max_total_draft_tokens=draft_len
    )
    md.runtime_draft_len = draft_len
    md.vocab_size = vocab
    md.is_all_greedy_sample = False
    md.advanced_sampling_mode = mode
    md.use_rejection_sampling = True
    md.prepare_rejection_sampling_buffers()
    rows = num_gens * (draft_len + 1)
    md.temperatures = torch.full((rows,), temperature, device="cuda")
    md.top_ks = torch.full((rows,), 2**31 - 1, dtype=torch.int32, device="cuda")
    md.min_ps = torch.zeros(rows, device="cuda")
    md.top_ps = torch.ones(rows, device="cuda")
    md.request_temperatures = torch.full((num_gens,), temperature, device="cuda")
    md.request_top_ks = torch.full((num_gens,), 2**31 - 1, dtype=torch.int32, device="cuda")
    md.request_min_ps = torch.zeros(num_gens, device="cuda")
    md.request_top_ps = torch.ones(num_gens, device="cuda")
    md.request_seeds = torch.arange(num_gens, dtype=torch.int64, device="cuda") + 1234
    md.request_offsets = torch.arange(num_gens, dtype=torch.int64, device="cuda") * 256 + 64
    md.seeds = md.request_seeds.repeat_interleave(draft_len + 1)
    md.offsets = md.request_offsets.repeat_interleave(draft_len + 1)
    md.batch_slot_ids[:num_gens] = torch.arange(num_gens, device="cuda")
    md.draft_probs_last_dim = vocab
    md.draft_probs.zero_()
    return md


def make_logits(dist: str, rows: int, vocab: int, gen: torch.Generator) -> torch.Tensor:
    """Synthetic fp32 logits ``[rows, vocab]`` of shape ``dist``.

    Each row puts the tokens in its own random order and gives the token of rank ``n``
    (1-based) the logit ``-alpha * ln(n)`` plus small noise. ``flat`` rows are standard
    Gaussian.
    """
    out = torch.empty((rows, vocab), device="cuda")
    alpha = DISTS[dist]
    if alpha is None:
        return out.normal_(generator=gen)
    perm = torch.randperm(vocab, generator=gen, device="cuda")
    base = -alpha * torch.log1p(torch.arange(vocab, device="cuda", dtype=torch.float32))
    for start in range(0, rows, 256):
        end = min(start + 256, rows)
        shift = torch.randint(0, vocab, (end - start, 1), generator=gen, device="cuda")
        out[start:end] = base[(perm.unsqueeze(0) + shift) % vocab]
    return out.add_(torch.randn(out.shape, generator=gen, device="cuda"), alpha=0.1)


def tokens_for_mass(logits: torch.Tensor, mass: float = 0.95) -> int:
    """Median number of most likely tokens that hold ``mass`` of the probability at temperature 1.

    The median runs over the first 32 rows of ``logits``.
    """
    probs = torch.softmax(logits[:32].double(), dim=-1).sort(dim=-1, descending=True).values
    kept = (probs.cumsum(dim=-1) - probs < mass).sum(dim=-1)
    return int(kept.median())


def time_eager(fn: Callable[[], object], warmup: int, iters: int) -> float:
    """Median latency of one eager call, in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))


def time_graph(fn: Callable[[], object], warmup: int, iters: int) -> float:
    """Median latency of one CUDA-graph replay of ``fn``, in microseconds.

    The warmup runs on a side stream first, so it compiles the kernels and grows the scratch
    buffers before the capture.
    """
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
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    us = statistics.median(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))
    del graph
    return us


@dataclass
class Result:
    mode: str
    dist: str
    temperature: float
    batch: int
    draft_len: int
    step: str  # "draft" or "verify"
    ref_graph_us: float
    fused_graph_us: float
    ref_eager_us: float
    fused_eager_us: float


def run_case(
    mode_name: str,
    dist: str,
    temperature: float,
    num_gens: int,
    draft_len: int,
    args: argparse.Namespace,
) -> list[Result]:
    """Times the draft and the verify step of one case on the reference and the fused path."""
    vocab = args.vocab
    mode = MODES[mode_name]
    gen = torch.Generator(device="cuda").manual_seed(num_gens * 100 + draft_len)
    target = make_logits(dist, num_gens * (draft_len + 1), vocab, gen)
    noise = torch.randn((num_gens, draft_len, vocab), generator=gen, device="cuda")
    drafts_in = target.view(num_gens, draft_len + 1, vocab)[:, :draft_len] + 0.3 * noise
    del noise
    times: dict[tuple[str, str], tuple[float, float]] = {}
    for impl in IMPLS:
        md = make_metadata(num_gens, draft_len, vocab, mode, temperature)
        worker = _Worker(draft_len)
        if impl == "ref":
            worker._fast_rs_applies = lambda *args, **kwargs: False
        flat_drafts = drafts_in.reshape(-1, vocab)
        for site, logits in (("draft", flat_drafts), ("verify", target)):
            fused = worker._fast_rs_applies(md, logits, vocab, site=site)
            assert fused == (impl == "fused"), f"{impl}: fused path {fused} at the {site} step"

        def draft_step(worker=worker, md=md):
            return worker.sample_draft_tokens(drafts_in, md, num_gens, num_contexts=0)

        draft_tokens = draft_step().view(num_gens, draft_len).clone()
        md.draft_tokens = draft_tokens.reshape(-1)
        # Both paths must run rejection sampling, not the exact-match fallback.
        assert worker._can_use_rejection_sampling(md) and worker._rejection_buffers_valid(
            draft_tokens, draft_len, vocab, 0, num_gens, target, md
        ), "the verify step would fall back to exact-match acceptance"

        def verify_step(worker=worker, md=md, draft_tokens=draft_tokens):
            return worker._accept_draft_tokens(target, draft_tokens, 0, num_gens, md)

        for step, fn in (("draft", draft_step), ("verify", verify_step)):
            graph_us = time_graph(fn, args.warmup, args.iters)
            eager_us = time_eager(fn, args.warmup, args.iters)
            times[(impl, step)] = (graph_us, eager_us)
        del md, worker
        torch.cuda.empty_cache()
    return [
        Result(
            mode_name,
            dist,
            temperature,
            num_gens,
            draft_len,
            step,
            times[("ref", step)][0],
            times[("fused", step)][0],
            times[("ref", step)][1],
            times[("fused", step)][1],
        )
        for step in ("draft", "verify")
    ]


def print_tables(results: list[Result], batch_sizes: list[int], draft_lens: list[int]) -> None:
    """Prints the speedup of the fused path per mode, logit shape and temperature, B by K.

    Each cell is the reference time over the fused time of the draft and verify steps together,
    as CUDA-graph replays: above 1 means the fused path is faster.
    """
    by_key = {(r.mode, r.dist, r.temperature, r.batch, r.draft_len, r.step): r for r in results}
    for mode, dist, temp in dict.fromkeys((r.mode, r.dist, r.temperature) for r in results):
        print(
            f"\n{mode}, {dist}, temperature {temp}: speedup of draft + verify, "
            "CUDA graph (ref / fused)"
        )
        print("B \\ K " + "".join(f"{k:>7}" for k in draft_lens))
        for b in batch_sizes:
            cells = []
            for k in draft_lens:
                pair = [by_key.get((mode, dist, temp, b, k, s)) for s in ("draft", "verify")]
                if None in pair:
                    cells.append(f"{'-':>7}")
                    continue
                ref = sum(r.ref_graph_us for r in pair)
                fused = sum(r.fused_graph_us for r in pair)
                cells.append(f"{ref / fused:>7.2f}")
            print(f"{b:>5} " + "".join(cells))


def print_slower(results: list[Result], tolerance: float) -> int:
    """Prints and counts the steps where the fused path is slower than the reference.

    A step counts when the fused path takes more than ``1 + tolerance`` times as long.
    """
    slower = []
    for r in results:
        for kind, ref, fused in (
            ("graph", r.ref_graph_us, r.fused_graph_us),
            ("eager", r.ref_eager_us, r.fused_eager_us),
        ):
            if fused > ref * (1 + tolerance):
                slower.append((r, kind, ref, fused))
    print(f"\nSteps where the fused path is more than {tolerance:.0%} slower: {len(slower)}")
    for r, kind, ref, fused in slower:
        print(
            f"  {r.mode:<16} {r.dist:<9} T={r.temperature:<4} B={r.batch:<4} K={r.draft_len:<3}"
            f"{r.step:<6} {kind}: ref {ref:9.1f} us, fused {fused:9.1f} us ({fused / ref:.2f}x)"
        )
    for kind in ("graph", "eager"):
        worst = max(
            results, key=lambda r: getattr(r, f"fused_{kind}_us") / getattr(r, f"ref_{kind}_us")
        )
        ratio = getattr(worst, f"fused_{kind}_us") / getattr(worst, f"ref_{kind}_us")
        print(
            f"Worst {kind} ratio fused / ref: {ratio:.2f} ({worst.mode}, {worst.dist}, "
            f"T={worst.temperature}, B={worst.batch}, K={worst.draft_len}, {worst.step})"
        )
    return len(slower)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--draft-lens", type=int, nargs="+", default=DEFAULT_DRAFT_LENS)
    parser.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--dists", nargs="+", default=list(DISTS), choices=list(DISTS))
    parser.add_argument("--vocab", type=int, default=KIMI_K3_VOCAB)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--tolerance", type=float, default=0.03, help="slowdown reported as slower (0.03 = 3%%)"
    )
    parser.add_argument("--save", default=None, help="write the results as JSON after every case")
    parser.add_argument(
        "--load", nargs="+", default=None, help="print the tables of saved JSON files instead"
    )
    args = parser.parse_args()

    if args.load:
        results = []
        for path in args.load:
            with open(path) as f:
                results += [Result(**r) for r in json.load(f)["results"]]
        print_tables(
            results, sorted({r.batch for r in results}), sorted({r.draft_len for r in results})
        )
        return 1 if print_slower(results, args.tolerance) else 0

    logger.set_level("warning")
    gpu = torch.cuda.get_device_name()
    print(f"GPU: {gpu}, vocabulary {args.vocab}")
    gen = torch.Generator(device="cuda").manual_seed(0)
    for dist in args.dists:
        sample = make_logits(dist, 32, args.vocab, gen)
        print(f"{dist:<9} 95% of the probability in {tokens_for_mass(sample)} tokens (median)")
    results: list[Result] = []
    for mode in args.modes:
        for dist in args.dists:
            for temp in args.temperatures:
                for b in args.batch_sizes:
                    for k in args.draft_lens:
                        case = run_case(mode, dist, temp, b, k, args)
                        results += case
                        d, v = case
                        print(
                            f"{mode:<16} {dist:<9} T={temp:<4} B={b:<4} K={k:<3} "
                            f"draft {d.ref_graph_us:8.1f} -> {d.fused_graph_us:8.1f} us  "
                            f"verify {v.ref_graph_us:8.1f} -> {v.fused_graph_us:8.1f} us",
                            flush=True,
                        )
                        if args.save:
                            with open(args.save, "w") as f:
                                out = {"gpu": gpu, "args": vars(args)}
                                results_json = [asdict(r) for r in results]
                                json.dump(out | {"results": results_json}, f, indent=1)
    print_tables(results, args.batch_sizes, args.draft_lens)
    return 1 if print_slower(results, args.tolerance) else 0


if __name__ == "__main__":
    raise SystemExit(main())
