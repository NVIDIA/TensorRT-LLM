---
name: trtllm-modeling-v2-perf-tuning
description: Load for any performance work on a modeling_v2 target -- the tune stage of the trtllm_modeling_v2 workflow. Carries the campaign discipline (ground truth, noise floor, what the keep rule judges, reference curves, gap decomposition), the kernel sourcing ladder for perf (trtllm op / in-tree module / FlashInfer installed version / torch / Triton / CuTe DSL / CUDA), the candidate protocol (worktree, entry + GPU test, micro-bench, decision file), and how to profile through the spool substrate.
---

# trtllm-modeling-v2-perf-tuning

A target that passed onboarding is *correct*. This skill makes it *fast* without
making it less correct. **Perf is measured, never gated -- but every kept change
passes the accuracy gate.** A campaign never publishes a number it did not
measure, and every number traces to a label dir.

## The layers

The serving config is fixed before any candidate: one per perf config,
derived from its workload (`tune/serving_config.py`) unless the spec names
one. With several perf configs (`perf.configs`) every candidate is measured at
all of them in one job.

| layer | who | what moves | judged by |
|---|---|---|---|
| **modeling iterations** | the `tuner` role, one candidate per turn | the target's forward: fusion, ordering, overlap, kernel source | micro-bench first (tuner), then e2e A/B + accuracy (orchestrator) |
| **runtime / native** | tuner flags `NEED_RUNTIME` / `NEED_REBUILD` | the engine itself (minimal edits only) | same as modeling, plus a rebuild |


## Ground truth before anything

1. `baseline` = target (`modeling_v2: require`) at the identity config.
2. `trtllm` = stock in-tree model (`modeling_v2: "off"`), **same config, same build, same workload**.
3. **noise floor** = at least three repeats of identical configs. Record the spread per headline
   metric. Cross-node repeats count only if they agree with same-node ones -- check, do not assume.
4. Profiles: one nsys capture per headline point (see [references/nsys.md](references/nsys.md)).

`modeling_v2: auto` is refused by the harness: a configuration that misses the target's routing
silently serves the stock model, and the number reads as the target's.

## The keep rule (applied by the orchestrator)

Judged per perf config on tok/s/user at the lowest concurrency and output tok/s/GPU at the highest
(tok/s/user at the highest as a regression check), against the mean of the same idle node's
controls (A before the candidate, A' after it) under the best tree's own measured noise floor:

- a gain above the noise-scaled threshold somewhere, and a loss beyond the noise **nowhere** -- a
  win on one perf config and a loss on another is a loss: split the idea or drop it;
- no latency constraint broken (`constraint.tpot_ms_max` / `ttft_ms_max`) that the control kept;
- then the accuracy gate. A perf win that fails accuracy is reverted.
- a loss within twice its floor is measured once more on the same node (B' then A'') before it
  counts; a member where the candidate is inert (its `path_marker` absent, its engine identical to
  the control's, or outside its `changed_members`) is not judged, and a loss there is host noise.

A hand-run A/B (a landing task, say) follows the same rule: A, B, A' on one idle node, judged with
`trtllm-modeling-v2 perf ab --a <A> --b <B> --a <A'> [--changed-members <configs>] [--path-marker
<text> [--marker-log <log>]]`. A round of only members outside the change judges none whose engine
reads as the control's when `--changed-members` names where it runs, or when the marker printed in
a `--marker-log` (a gate's log at a member that runs it).

Aim a candidate at a gain that survives every config, and say in the decision which configs you
expect it to move.

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

- Formatting and comment polish are not required on candidate commits (the operator lints the branch once, before it becomes a PR).
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
`trtllm-modeling-v2-core` (`references/graph-and-sync.md`) before editing a forward.

## Reference curves and the gap

| line | what | gated |
|---|---|---|
| `trtllm` | stock modeling + identity | no |
| `baseline` | target + identity | yes (onboarding) |
| `iter-<slug>` | target + identity + kept overlays + kept modeling commits | yes, every keep |

Both run at the same serving config, so best / trtllm is the modeling delta at an identical config.

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
| the target's own switches | an env var or a dormant capability of the target is a candidate like any other: one hypothesis, measured e2e | tuner turns |
| the profile | nsys per step class, ranked by exclusive time a change can remove; refreshed on the moved best tree when the roadmap is spent | roadmap turns -> `roadmap*.yaml` |
| fusion (memory round trips) | a producer/consumer pair that writes and re-reads an activation: fold it into one kernel or a GEMM epilogue ([references/modeling-iterations.md](references/modeling-iterations.md) "Fusion patterns") | tuner turns |
| every hot kernel, four questions | agent-flow perf_optimize's kernel ledger: each kernel down to the coverage target (`campaign.kernel_coverage_pct`, default 80% of GPU time) answers elimination / faster / fusion / overlap with a roadmap slug or an evidenced dismissal -- so overlap and reordering are asked of every kernel, not only when someone thinks of them | roadmap turns write `<roadmap>-kernel_ledger.yaml`; the stage validates it with perf_optimize's `load_ledger` + `cross_validate` and gives one fix turn (`campaign.kernel_ledger: require \| off`) |
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
the gap to `trtllm` is `closed`, `untried`, `infeasible: <why>` or `unexplained`.

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
