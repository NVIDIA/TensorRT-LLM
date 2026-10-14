---
name: staircase-perf-tuning
description: Load for any performance work on a modeling_v2 target -- the tune stage of the staircase onboarding workflow. Carries the campaign discipline (ground truth, noise floor, keep rule, reference curves, gap decomposition), the split between the Python config ladder and agent modeling iterations, the kernel sourcing ladder for perf (trtllm op / in-tree module / FlashInfer installed version / torch / Triton / CuTe DSL / CUDA), the candidate protocol (worktree, entry + GPU test, micro-bench, decision file), and how to profile through the spool substrate.
---

# staircase-perf-tuning

A target that passed onboarding is *correct*. This skill makes it *fast* without
making it less correct. **Perf is measured, never gated -- but every kept change
passes the accuracy gate.** A campaign never publishes a number it did not
measure, and every number traces to a label dir.

## The three layers, cheapest first

| layer | who | what moves | judged by |
|---|---|---|---|
| **config ladder** | Python (`tune/ladder.py`) | LLM-API runtime knobs and the target's own env switches | the keep rule, automatically |
| **modeling iterations** | the `tuner` role, one candidate per turn | the target's forward: fusion, ordering, overlap, kernel source | micro-bench first (tuner), then e2e A/B + accuracy (orchestrator) |
| **runtime / native** | tuner flags `NEED_RUNTIME` / `NEED_REBUILD` | the engine itself (minimal edits only) | same as modeling, plus a rebuild |

The ladder runs first because it sets the configuration every modeling change is
judged at. A modeling change measured at a config the ladder later moves is
re-measured, not extrapolated.

## Ground truth before anything

1. `baseline` = target (`modeling_v2: require`) at the identity config.
2. `trtllm` = stock in-tree model (`modeling_v2: "off"`), **same config, same build, same workload**.
3. **noise floor** = at least three repeats of identical configs. Record the spread per headline
   metric. Cross-node repeats count only if they agree with same-node ones -- check, do not assume.
4. Profiles: one nsys capture per headline point (see [references/nsys.md](references/nsys.md)).

`modeling_v2: auto` is refused by the harness: a configuration that misses the target's routing
silently serves the stock model, and the number reads as the target's.

## The keep rule

For every measured point, `delta = (candidate - control) / control` on the judged metric
(tok/s/user at the lowest concurrency, output tok/s/GPU at the highest, tok/s/user at the highest
too, as a regression check). With `noise[m]` the measured spread:

- **WIN**: some delta >= `max(min_gain, noise_factor * noise[m])`, and no delta <= `-max(1%, noise_factor * noise[m])`.
- **LOSS**: some delta below that floor -- even if another point won. Split the idea or drop it.
- **NEUTRAL**: everything inside the band. Neutral is not a keep.

Then the accuracy gate (`modeling_v2: require`, identity + every kept overlay). A perf win that
fails accuracy is reverted. Never re-run a gate until it passes.

## The candidate protocol (modeling iterations)

One turn = one hypothesis = one candidate:

1. **Evidence.** Name the kernels/ops and their per-step time from a profile, or the op count from
   code reading, that motivate the hypothesis. Rank by *exclusive* time a change can remove.
2. **Worktree.** Work only in the worktree the request names (`tune/<slug>` branch, native build
   linked in). Never edit the main checkout.
3. **Implement within the contract.** A new op is a new entry: an `OpWrapper` with `__call__`,
   `reference` (float64 or the op it replaces), `is_valid`, `CELLS` at the shapes the forward
   actually reaches (decode batch 1..max_con, prefill chunk up to `max_num_tokens`), `ARCHS`, `note`,
   and a GPU test that names `.CELLS`. It lives in the target's `ops/<category>/` unless a second
   target calls it. See [references/modeling-iterations.md](references/modeling-iterations.md).
4. **Correctness.** The entry's GPU test passes at every cell, and the target's existing tests for
   the ops it replaces still pass (or are replaced by equivalent cells).
5. **Micro-bench** with `tune/microbench.py`: the current implementation vs the candidate at decode
   M = 1 and M = max_con, and at a prefill chunk, under CUDA graphs and eager, with per-layer
   weight rotation. Record every alternative source you timed.
6. **Decision file** `CANDIDATE` with the evidence, the commits, the micro-bench table and the
   *expected* e2e effect. The orchestrator measures e2e and runs the gate; you never claim an e2e
   number you did not read from a label dir.

A hypothesis that does not survive the micro-bench is `DROPPED` with its numbers -- negative
results steer the next turn and go into the campaign report.

## Kernel sourcing for perf

Cheapest first; **when several sources serve one op, micro-bench them all, keep the fastest, and
record the rest** in `kernel_choices.json` (op, workload, chosen, alternatives with their times):
a different workload may pick a different one.

| tier | source | notes |
|---|---|---|
| 1 | `torch.ops.trtllm.*` | check the op's dtype/shape support at the cells you need |
| 2 | in-tree callables under `tensorrt_llm/_torch/modules/**`, `custom_ops/**` | read to learn the interface, never to copy a model's structure |
| 3 | FlashInfer public API **as installed** (`python -c "import flashinfer; print(flashinfer.__version__)"`) | a kernel that exists only in a newer FlashInfer is not available -- record it as an upgrade candidate |
| 4 | native torch | composition is fine when it fuses nothing away |
| 5 | Triton written by this run | fused elementwise/norm/epilogue chains, small-M GEMM + epilogue |
| 6 | CuTe DSL / CUDA written by this run | only when 1-5 cannot reach the bandwidth or launch floor; say why |

Model-specific kernels of an **embargoed** in-tree implementation are not a source (the task's
`source_embargo` lists them; `torch.ops.trtllm.<model>_*` registered by those files included).

## What the harness relaxes for a perf campaign

- Formatting and comment polish are not required on candidate commits (the pre-merge pack lints later).
- The flat-forward rule stands, but a *fused entry* may span several former op boundaries.
- Bit-identity with the previous forward is **not** required; entry-level numerics within the
  entry's band against the fp64 reference, plus the accuracy gate, are the criteria.
- A runtime edit is allowed when the lever is runtime-owned and no target-side alternative exists:
  keep it minimal, flag it `NEED_RUNTIME`, and say what else it could affect.

## Graph safety (decode runs under CUDA graphs)

No host synchronization in the decode path (`.item()`, `.cpu()`, boolean indexing, `torch.nonzero`,
data-dependent Python branches). Host ints the engine fixes per capture are fine. Every buffer a
kernel writes must be allocated inside the forward or owned by the target, never a fresh tensor
captured once and reused across shapes. Side streams: record/wait events, capture-safe. Load
`staircase-modeling-core` (`references/graph-and-sync.md`) before editing a forward.

## Reference curves and the gap

| line | what | gated |
|---|---|---|
| `trtllm` | stock modeling + identity | no |
| `trtllm-tuned` | stock + identity + the ladder's runtime overlay | no |
| `baseline` | target + identity | yes (onboarding) |
| `iter-<slug>` | target + identity + kept overlays + kept modeling commits | yes, every keep |

Decompose: end-to-end = best / trtllm; portable config = trtllm-tuned / trtllm; modeling delta at
identical config = best / trtllm-tuned.

## Profiling

Through the substrate only: `bench.py --nsys` inside a spool job. Session mode is the default (`nsys
launch` + an external start/stop window after `--nsys-delay`); TensorRT-LLM's in-process recipe
(`TLLM_PROFILE_START_STOP=A-B`, `-c cudaProfilerApi`) is `--nsys-mode api`. Summarize with
`tune/nsys_summary.py` per step class (decode / mixed / prefill): at high concurrency about half the
run is mixed (prefill) steps, which a decode-only profile hides. Flags and extraction recipes:
[references/nsys.md](references/nsys.md). Never quote a profiled run's throughput.

## Where hypotheses come from

Look in this order; each source has a place in the stage. Roadmap items and decision files carry a
`sources` list (`{source, ref, outcome}` with `source` one of `profile`, `code-reading`, `flashinfer`,
`repo-prior-art`, `sister-engine`, `kernel-choices`); the campaign ledger keeps it per candidate, so the
next campaign can see which sources paid off.

| source | how | where it lives in the stage |
|---|---|---|
| runtime knobs and the target's own switches | the Python config ladder; a target switch (an env var, a dormant capability) is an `extra_rungs` entry | `tune/ladder.py`, `perf.ladder.extra_rungs` |
| the profile | nsys per step class, ranked by exclusive time a change can remove; refreshed on the moved best tree when the roadmap is spent | roadmap turns -> `roadmap*.yaml` |
| fusion (memory round trips) | a producer/consumer pair that writes and re-reads an activation: fold it into one kernel or a GEMM epilogue ([references/modeling-iterations.md](references/modeling-iterations.md) "Fusion patterns") | tuner turns |
| ordering and overlap | independent work on a side stream; PDL between dependent small kernels; hoist per-step invariants (same reference, "Ordering") | tuner turns |
| FlashInfer as installed | the preflight's `surface.md` "FlashInfer API" lists the installed version's callables the engine does not call -- a tier-3 entry (`from flashinfer import ...`) in the shared `catalog/<category>/`, raced against the alternatives | request "Sources to check", `kernel_choices.json`, `modeling_v2/catalog/` |
| prior art | this repository's perf commits on similar modules (`git log -i --grep=perf -- <module>`), and the sister engines' strategies (vLLM, SGLang paths in the request) -- strategies only, never the embargoed in-tree implementation of this model | request "Sources to check", `perf.sister_engines` |
| earlier campaigns | `kernel_choices.json` records the chosen kernel *and* the losing alternatives with timings: a new workload re-reads it before racing again | `<workdir>/08-kernel-sources/kernel_choices.json` |

Micro-bench first, e2e second: a tuner proves a hypothesis on the op or block (`tune/microbench.py`,
graph and eager timings, cold weights) before the orchestrator spends a node on it, and several
winners are stacked and measured once.

## Stop

Stop when three consecutive candidates were not kept **and** no profiled item can remove more than
the keep threshold, or when the budget is spent. A stop carries a remaining-gap table: every part of
the gap to `trtllm-tuned` is `closed`, `untried`, `infeasible: <why>` or `unexplained`.

## Red flags

| impulse | reality |
|---|---|
| "this should be ~2x faster" | Measure it. Micro-bench numbers in the decision, e2e from the orchestrator. |
| "the stock model is fast here, let me see how it does it" | Its model-specific files are embargoed. Shared modules and kernels are fair game; its forward is not. |
| "two fusions, one candidate -- saves a sweep" | One hypothesis per candidate, or attribution is garbage. Batch only with a leave-one-out table. |
| "padding/knob X won elsewhere" | Priors are dated observations; this target re-measures. |
| "a newer FlashInfer has the kernel" | Not available until the pin moves. Record it as an upgrade candidate. |
| "the test band is too tight, widen it" | The band is the entry's contract. Fix the kernel, or record why the reference is the wrong one. |
| "skip the gate, it is just a fusion" | Fusions change rounding. Every keep passes the gate. |
