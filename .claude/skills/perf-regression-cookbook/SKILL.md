---
name: perf-regression-cookbook
description: >
  Cookbook of how TensorRT-LLM performance regressions happen and how they
  were fixed, distilled from NVBug-fix commits on main. Consult when a perf
  regression is observed or suspected (throughput/TTFT/ITL drop, perf CI bar
  failure, memory-footprint growth) to match the symptom against known
  regression patterns and their precedent fixes. Each pattern precisely
  tracks all related commit hashes and NVBug IDs. Also consult when
  reviewing a risky change to check which regression classes it could
  reintroduce.
tags:
  - regression
  - cookbook
  - diagnostics
  - decision-support
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Performance Regression Cookbook

A curated library of **TensorRT-LLM performance regressions and their fixes**,
recorded as diagnostic precedents. Use it to answer "*this* metric regressed —
what usually breaks it, where do I look first, and how was this class fixed
before?" — instead of re-deriving the failure mode from scratch.

This is the defensive twin of `perf-optimization-casebook`: that skill records
how to make things *faster*; this one records how things *got slower* and how
they were repaired. Every case is anchored to NVBug ID(s) + fix commit(s) +
PR(s), so any case can be deep-dived later through its bug ID.

This skill is **reference material, not a coordinator.** It does not run
profiling or edit code. It is consulted by `perf-analysis` when a regression
(not a plain bottleneck) is being diagnosed, by reviewers assessing the
regression risk of a change, and by agents triaging perf CI bar failures.

## When to Consult

- A perf metric regressed vs a previous build/release (TPS, TTFT, ITL/TPOT,
  memory→achievable batch, startup time) and you need candidate failure modes
  ranked by prior frequency.
- A perf CI bar / QA sweep tripped and you are triaging the offending commit
  range.
- You are reviewing or writing a change to a perf-sensitive area (fusion
  patterns, CUDA-graph regions, scheduler, autotuner, kernel selection,
  dependency bumps) and want the known ways such changes have regressed perf.
- You just fixed a perf regression and want to record it for future reuse.

## How to Use — the loop

1. **Characterize.** Which metric regressed, on what model/hardware/phase, and
   what changed in between (commit range, dependency bump, config change). If
   you have a profile, note the signature (new sync? unfused kernel stream?
   different kernel name? host span growth?).
2. **Match.** Normalize your terms via `data/aliases.yaml`, match a
   **regression pattern** in `data/patterns.yaml` (symptom signals →
   pattern), then grep case frontmatter for the pattern id or canonical
   terms and open only the winning case file(s):

       grep -l "pattern-<id>" references/*/*.md
       grep -l "fusion-broken" references/*/*.md

3. **Localize.** Use the matched cases' **Detection signal** to confirm or
   refute the class on your setup (grep the log marker, check the knob,
   look for the profile signature) before bisecting blindly. Prior instances'
   **How introduced** tells you which kinds of commits to suspect first in
   the range.
4. **Fix & verify.** Adapt the precedent's **Fix mechanism**; measure the
   metric that regressed. Numbers come from real measurement only.
5. **Record.** Add or update a case (see "Adding a New Case"); regressions
   recur in classes, and the next triage should start warmer.

### Consultation trace (machine-parseable — always emit)

Every consultation MUST end with exactly ONE trace line in your visible reply
text, as soon as the Match step resolves:

    REGRESSION-COOKBOOK: matched=<family>/<case-slug> confidence=<high|medium|low> adapted="<one-line adaptation>"

or, when no precedent fits:

    REGRESSION-COOKBOOK: no-match query="<regression class / signals you searched for>"

`<case-slug>` is the case file name without `.md`. Emit the line even when the
consultation is a dead end: it is the join key for automated audit trails, and
the no-match rate is the cookbook's coverage metric for maintainers.

## Corpus & provenance

- Source: TensorRT-LLM `main`, NVBug-titled commits
  (`[https://nvbugs/<id>][fix|perf|...]`), window starting 2026-01-10.
- Current corpus (2026-07-10 commit scan + 2026-08-11 NVBug-first sweep,
  per-case evidence audit 2026-08-12, coverage backfill 2026-08-12): 725
  nvbug-titled commits scanned → 637 candidates classified from PR descriptions
  → **60 confirmed cases** (70 commits, 74 NVBugs) + **25 pending** suspected
  entries.
- **The last +17 came from asking the corpus what it was missing, not from a
  new scan.** A cross-check of every performance-severity NVBug with a
  merged fix PR against the case list produced 24 uncovered bugs, which folded
  into 17 new regression cases (plus one instability case) — i.e. roughly a
  quarter of the eligible population was absent *after* two sweeps and an
  evidence audit had each declared the corpus complete. Both sweeps were
  discovery-shaped (scan a source, classify what it yields); neither could see
  a bug it never enumerated. The cheap standing check is therefore the
  *inverse*: enumerate eligible bugs from NVBugs and diff against
  `nvbugs:` frontmatter across all cases. Do that before adding another scan.
  The 2026-08-12 audit had moved the count from 47 to 43: six cases were
  removed as functional bugs
  (see `data/removed.yaml` — they are refuted, *not* pending), one
  (`mla-chunked-prefill-maybe-compiled-cat-warmup`) arrived from the
  instability cookbook, and one (`ipc-hmac-key-via-fd-breaks-bench`) was added
  by the same-day PR-coverage cross-check — it had been cited as a *confounder*
  in three existing cases before anyone gave it a case of its own, which is the
  cheapest available signal that a case is missing. Every case carries at least one NVBug, and
  every case has a **merged** fix on `main` — a bug whose fix has not landed
  is not recorded here at all. One umbrella perf bug (5615248) is split
  into three cases by root cause; conversely one case folds **five** NVBugs
  (`deepgemm-pdl-import-time-cuda-context`:
  6419139+6418453+6419078+6390244+6402018, one import-scope defect reported
  five times).
- **Folding has a limit, and it is the bug's own severity.** An earlier
  revision folded the whole trtllm-gen FMHA warmup family into one case
  (6185446+6193854+6248837+6293823+6315845). That was wrong twice over:
  6248837 is a functional bug, not a perf bug at all, and the family is two
  *distinct* defects fixed by two PRs a month apart — no warmup (#14851) and
  then a too-sparse warmup grid (#15305). It is now
  [round 1](references/jit-and-warmup/trtllmgen-fmha-jit-warmup.md) +
  [round 2](references/jit-and-warmup/trtllmgen-fmha-densify-grid.md),
  cross-linked. Rule: fold only bug ids that share **one root cause and one
  fix PR**; check the severity on every id before folding, and
  split when the same failure mode needed a second independent fix — the
  second fix's *failed attempts* are the transferable knowledge, and folding
  erases them.
- The second sweep was **NVBug-first, not commit-first**: NVBugs with
  performance severity and `regression`/`unstable`/`flutter` in the
  title, resolved to a fix PR per bug. That is what found the fixes whose
  commit subject carries no nvbug tag (e.g. `[None][feat]`-titled PRs) and
  the *failed* attempts, which a commit scan cannot see at all.

### What counts as a "regression" here — three natures

All cases are confirmed **perf-bug fixes**, but NVBug perf bugs come in three
natures. Natures 1 and 2 partition the corpus by the `introduced_via:`
frontmatter field (39 + 19 + 2 `unknown` = 60); nature 3 is a `regression_class`
and therefore **overlaps** both — a measurement artifact still got introduced
somehow, so those 7 cases are counted twice here on purpose.

1. **True regressions** (39 cases) — something was faster before and a
   change made it slower: `prior-fix-side-effect`, `new-feature`,
   `kernel-change`, `config-default-change`, `dep-bump`, `refactor`,
   `incomplete-coverage` (an optimization shipped but a path/dtype/shape it
   missed lands on the slow path).
2. **Below-expectation perf bugs** (19 cases) — never was fast; NVBug filed
   because perf was under the bar: `introduced_via: [pre-existing-gap]`.
   Not regressions strictly, but the same failure modes recur as regressions,
   so they stay in the match surface.
3. **Measurement artifacts** (7 cases) — the *measured number* moved without
   the compute path changing (`measurement-artifact` class; six live in the
   measurement-and-test family, one in communication). Usually the test config
   changed and the product did not — but not always: in
   `ipc-hmac-key-via-fd-breaks-bench` a real product change (fd-based IPC
   handshake) deadlocked server bring-up, so the bar tripped with a product
   commit in range that no profile would implicate. "Measurement artifact"
   means the compute path is innocent, **not** that the product is.

Those counts are **mechanical**, so a later sweep can reproduce them: nature 3
is `regression_class:` containing `measurement-artifact`; of the rest, nature 2
is `introduced_via:` containing `pre-existing-gap` and nature 1 is everything
else. (Two of the 12 `pre-existing-gap` cases are also measurement artifacts
and are counted once, in nature 3.)

When triaging a **known-good → bad transition** (you have a commit range),
prioritize nature 1 — filter out the never-was-fast cases first:

    grep -L "pre-existing-gap" references/*/*.md

When the baseline itself is in question ("is this as fast as it should
be?"), natures 2 and 3 are exactly the precedents to check.
- Every case carries `nvbugs:`, `commits:`, `success_prs:`, `failed_prs:`
  frontmatter — the complete provenance for that regression. Patterns aggregate
  provenance transitively through their `instances:` list.
- **`failed_prs:` is the half to read before proposing a fix.** It lists the
  attempts on the same NVBug that did *not* land — closed unmerged, superseded,
  or rejected on review — with the reason in the case's **Failed attempts**
  bullet. A plausible fix already tried and rejected is the single most
  expensive thing to re-propose. `success_prs:` is what merged.
- **An attempt still open is in neither list**, and a bug whose *only* attempts
  are open or failed has no case at all — the cookbook records NVBugs with a
  valid, merged fix PR, because a case's value is its citable **Fix mechanism**.
  So `failed_prs:` is always read in the context of a fix that did land: it says
  "this was tried on the way to the answer below", never "this bug is open".
- Verdicts were established from **PR descriptions + diffs only**; NVBug
  *history* was deliberately not read. Future deep-dives can start from any
  case's `nvbugs:` IDs.
- Commits whose PR description was too unclear to confirm but which are
  *suspected* perf-regression fixes are parked in **`data/pending.yaml`** —
  the standing to-classify pool. Do not cite pending entries as precedents;
  they are a deep-dive worklist. One entry (`6070875`) is *inverted* — the
  regression is fully established and the fix attribution is not; the same
  "do not cite" rule applies.
- **Scan cursors:** commit scan `fd9166c0a7` (origin/main, 2026-07-10);
  NVBug sweep covers ids above a recent floor at performance severity as of
  2026-08-11. Commit sweeps resume from the hash; dedup by grepping it across
  case `commits:` fields and `data/pending.yaml`. NVBug sweeps resume from the
  id floor; dedup by grepping the id across `nvbugs:` and `data/pending.yaml`.
  A sweep will re-encounter bugs it already rejected for having no merged fix,
  which is intended — the right question each time is whether one has landed
  since. Neither cursor subsumes the other — the NVBug sweep
  found fixes whose commit subject has no nvbug tag, and the commit scan finds
  fixes whose bug was never filed as a performance bug.

## Modules — routing table

`references/` is indexed by **module**: the piece of TRT-LLM the fix landed
in. Open the module **index** nearest your regression, then drill into a
single case file from its case table (each module is a directory of
one-file-per-case behind an `index.md` of patterns + a case-picker table).
The canonical id list, with a prose description of each module's code, is
`data/modules.yaml` — identical in the instability cookbook, so a module id
queries both corpora.

**Filing rule (and reading rule): the module is where the FIX landed, not
where the symptom showed.** A missing FMHA warmup grid is fixed by editing
the warmup enumeration → `jit-and-warmup`; a slow FMHA dispatcher is fixed in
the dispatcher → `attention-fmha`. Both look like "an attention problem" from
the bug report. If you are unsure which of two modules to open, use
`subsystems:` instead — it is multi-valued and crosses modules:
`grep -l 'attention-kernel' references/*/*.md`.

| Module | Index | Covers | Cases |
|--------|-------|--------|-------|
| Attention & FMHA | [references/attention-fmha/](references/attention-fmha/index.md) | FMHA / trtllm-gen attention, MLA, DSA sparse attention + indexer, and host-side kernel choice / input layout | 5 |
| Kernel fusion | [references/kernel-fusion/](references/kernel-fusion/index.md) | fusion patterns and the passes/graph shapes they need; gating that silently unfuses | 2 |
| JIT & warmup | [references/jit-and-warmup/](references/jit-and-warmup/index.md) | which shapes/dtypes/modes get compiled before serving; compile or autotune cost landing on a live request | 6 |
| CUDA graph & compile | [references/cuda-graph-and-compile/](references/cuda-graph-and-compile/index.md) | capture/replay coverage, batch buckets, eager-fallback gates, piecewise graphs, torch.compile regions | 5 |
| MoE | [references/moe/](references/moe/index.md) | routing, grouped/router GEMMs, expert dispatch/combine, a2a backends (DeepEP, MNNVL) and their gating | 4 |
| GEMM & quantization | [references/gemm-and-quantization/](references/gemm-and-quantization/index.md) | dense GEMM paths, quantized linear layers, FP8/FP4 quantize-dequantize, DeepGEMM/CUTLASS wrappers | 2 |
| Autotuner | [references/autotuner/](references/autotuner/index.md) | tactic / tunable-runner machinery: what is tuned, cache keys, when tuning runs, runtime dispatch | 1 |
| Communication | [references/communication/](references/communication/index.md) | AllReduce/AllGather, NCCL (incl. symmetric), MNNVL, user buffers, strategy selection and fast-transport fallbacks | 3 |
| KV-cache manager | [references/kv-cache-manager/](references/kv-cache-manager/index.md) | block pool + reuse accounting: capacity/size estimation, token budgets, per-dtype sizing, recurrent/SSM state | 4 |
| KV-cache transceiver | [references/kv-cache-transceiver/](references/kv-cache-transceiver/index.md) | moving cache ctx→gen: the transceiver, NIXL/UCX transfer, negotiation/quorum | 2 |
| Disagg orchestrator | [references/disagg-orchestrator/](references/disagg-orchestrator/index.md) | disagg front end: ctx/gen routing, admission + throttling, the orchestrator process | 2 |
| Scheduler & executor | [references/scheduler-and-executor/](references/scheduler-and-executor/index.md) | the PyExecutor loop: admission/batching, overlap scheduling, per-request bookkeeping, model-engine step | 3 |
| Sampler | [references/sampler/](references/sampler/index.md) | TorchSampler, beam search, logprobs, stop words, and their host handoff / D2H copies | 5 |
| Speculative decoding | [references/spec-decode/](references/spec-decode/index.md) | MTP, EAGLE, draft models, acceptance accounting, token-to-request bookkeeping | 2 |
| Tokenization & detokenization | [references/tokenization/](references/tokenization/index.md) | input processing and detokenization on the request path, per-model tokenizer fast/slow paths | 1 |
| Model definition | [references/model-definition/](references/model-definition/index.md) | per-model Python modeling: layer wiring, weight load/restore, in-model backend/dtype gates, vision towers | 3 |
| Runtime & serving | [references/runtime-and-serving/](references/runtime-and-serving/index.md) | trtllm-serve endpoints + media I/O, LLM-API plumbing, buffer pools, custom-op dispatch, capability detection | 4 |
| Measurement & test | [references/measurement-and-test/](references/measurement-and-test/index.md) | the number moved but the product did not: perf test configs, benchmark driver + parsers, test infra, container/dep conditions | 6 |
| Case schema + template | [references/case-template.md](references/case-template.md) | The field schema and rules for adding cases | — |

Five of these modules also hold cases in **perf-instability-cookbook**
(`jit-and-warmup`, `communication`, `moe`, `scheduler-and-executor`,
`measurement-and-test`) — check both when a regression's signature includes
run-to-run variance.

### Quick index — regression signature → where to look

Match the signature, open the pattern in `data/patterns.yaml`, then its
instance cases. Patterns are the **cross-module** axis: a pattern listing
many modules is the useful case, not a filing error — it says this mechanism
already recurred in code with nothing else in common.

| Observed signature | Pattern | Modules |
|---|---|---|
| Host span grew vs previous build; GPU waits on CPU; ITL/TPOT up | `pattern-host-work-on-hot-path` | 12 of the 18 — the widest pattern in the corpus; start at `attention-fmha`, `runtime-and-serving`, `sampler`, `scheduler-and-executor` |
| New per-iteration collective/vote or `.item()`/D2H after a correctness fix | `pattern-per-step-sync-added` | sampler, attention-fmha, kv-cache-manager, kv-cache-transceiver |
| Trace shows a different/slower kernel or NCCL instead of MNNVL/SYMM_MEM/NVLS; run is correct, just slow | `pattern-fast-path-silent-fallback` | cuda-graph-and-compile, kernel-fusion, model-definition, attention-fmha, communication, kv-cache-manager, moe, runtime-and-serving |
| A kernel change/optimization landed in the range; one dtype/shape regime got slower | `pattern-kernel-swap-regressed` | attention-fmha, gemm-and-quantization, moe, spec-decode |
| Sporadic multi-second stalls mid-run; first-iteration or narrow-shape slowdowns | `pattern-warmup-coverage-gap` | jit-and-warmup, measurement-and-test |
| Fewer KV blocks / smaller batches reported; kernels unchanged | `pattern-capacity-accounting-error` | kv-cache-manager |
| Perf bar tripped right after a defaults change | `pattern-default-change-regressed-tuned-workload` | cuda-graph-and-compile |
| Throughput ramps slowly / gaps at high concurrency after an admission change | `pattern-admission-throttle-misconfigured` | disagg-orchestrator |
| TTFT up under overlap scheduling, ITL unchanged | `pattern-delayed-first-token-emission` | scheduler-and-executor |
| Perf CI bar failure with no plausible product commit in range | `pattern-measurement-not-product` | measurement-and-test, communication |
| Bar tripped, but the run produced no output after server launch / workers idle at 0% GPU | `pattern-startup-handshake-fd-not-inherited` | measurement-and-test |
| Prefill cost scales with TP world size | `pattern-redundant-work-across-ranks` | attention-fmha |
| Branch/port slower than main on a path main already fixed | `pattern-optimization-lost-in-port` | sampler |
| Spec-decode acceptance length dropped after a config/validator change | `pattern-variant-misroute` | attention-fmha |
| Fused region reappears as many small kernels after a dep bump/refactor | `pattern-fusion-pattern-drift` | kernel-fusion |
| Host cost per step scales with a feature that is *disabled* for this run | `pattern-inactive-feature-guard-on-hot-path` | moe |
| KV pool / block count shrank; config and free memory unchanged; per-step GPU time flat | `pattern-unaccounted-startup-residency` | gemm-and-quantization, jit-and-warmup, model-definition |
| Regression appeared right after a graph/autotuner **bucket was added**, and only above the previous ceiling | `pattern-padding-bucket-overshoot` | cuda-graph-and-compile |
| Kernel perf held hostage by a pinned third-party dep; fix is a version bump | `pattern-pinned-dep-holds-kernel-perf` | moe |
| Buffers zero-filled or re-initialized when the consumer overwrites them anyway | `pattern-unneeded-buffer-initialization` | runtime-and-serving |
| A selection heuristic keys on one dimension of a multi-dimensional crossover, so a whole regime gets the wrong path | `pattern-selection-heuristic-too-coarse` | communication |
| A cached index/offset tensor encoding a token layout is reused after the layout changed | `pattern-stale-metadata-across-layout-change` | spec-decode |

## Case Schema

See `references/case-template.md`. Core fields: **Provenance** (nvbug ·
commit · PR — mandatory), **Symptom**, **Root cause**, **How introduced**,
**Fix mechanism**, **Detection signal**, **Prevention/guard**,
**Generalizes to**.

## Principles

1. **Trace, don't recall.** Every number, knob, and path must be traceable to
   the fix PR's description or diff. Regression sizes appear only when the
   PR/bug states them, with the source cited.
2. **Never present a precedent as a diagnosis.** A matched case is a
   hypothesis to *check* (via its Detection signal), not a conclusion.
3. **Patterns are the match surface.** Cases are instances; match on the
   pattern and the canonical signals, not on titles.
4. **Pending is not precedent.** `data/pending.yaml` entries are unconfirmed;
   promote them to cases only after their nature is established (e.g. a
   future bug-history deep-dive).
5. **Interface facts are as-of their pinned commit.** Knob names and paths in
   a case describe the code at the case's `commits:`; re-verify against your
   checkout before acting.

## Adding a New Case

1. Confirm it is a perf-regression fix (explicit evidence in PR/bug text —
   or your own measurement, for locally-found regressions).
2. Fold vs. new: a PR series fixing the *same* regression (same nvbug or same
   root cause) is ONE case with `related:` provenance lines. A new case needs
   a new root cause or a new pattern.
3. Pick the module by **where the fix landed** (`data/modules.yaml`); create
   `references/<module>/<slug>.md` from `references/case-template.md`; fill
   every frontmatter field with canonical terms from `data/tags.yaml` —
   `module:` must equal the parent directory name; register/extend the pattern
   in `data/patterns.yaml` (bidirectional: case `patterns:` ↔ pattern
   `instances:`, and add the module to the pattern's `modules:` list); add a
   row to that module's `index.md` case table.
4. Keep the anti-fabrication rules of the template. **A case means "a merged fix
   is on `main`"** — that is what makes its Fix-mechanism bullet citable. Two
   ways to fail that test, one destination each: a real fix whose *perf nature*
   is unconfirmed goes to `data/pending.yaml`; a confirmed regression with **no
   merged fix** is not recorded at all, however well understood it is. Do not
   add a case for it, and do not invent a parking lot for it — revisit the bug
   when a fix lands.
5. This cookbook ships in the public TensorRT-LLM repository. Cite an NVBug by
   id and restate what it established in your own words; never copy its title
   or comments, or name customers, people, status or priority values.

## Relationship to Other Skills

- **perf-optimization-casebook** — the offensive twin (how to get faster).
  Some regressions are optimizations that broke; cross-link the casebook case
  when one exists.
- **perf-analysis / perf-host-analysis** — classify and localize the
  bottleneck; this cookbook then maps regression signatures to failure modes.
- **trtllm-code-contribution** — the Prevention/guard field feeds review
  checklists for perf-sensitive changes.
