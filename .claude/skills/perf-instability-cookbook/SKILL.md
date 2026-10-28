---
name: perf-instability-cookbook
description: >
  Cookbook of how TensorRT-LLM performance *instabilities* happen and how they
  were mitigated, distilled from perf-instability commits on main. Consult when
  a metric is unstable across runs, iterations, or ranks (first-iter JIT stall
  spikes, cross-run throughput variance, non-deterministic hangs or logs, NCCL
  negotiation flakiness, perf-test read-after-write races) to match the symptom
  against known instability patterns and their precedent mitigations. Distinct
  from `perf-regression-cookbook` (known-good → bad transitions) and
  `perf-optimization-casebook` (offensive optimizations): here perf is *variable*
  rather than uniformly slower. Each pattern precisely tracks all related commit
  hashes and NVBug IDs. Also consult when reviewing a change that adds JIT
  compilation, opportunistic collectives, or GC-managed collective resources.
tags:
  - instability
  - cookbook
  - diagnostics
  - decision-support
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Performance Instability Cookbook

A curated library of **TensorRT-LLM performance instabilities and their
mitigations**, recorded as diagnostic precedents. Use it to answer "*this*
metric is unstable — what usually causes that class of instability, where do I
look first, and how was it mitigated before?" — instead of re-deriving the
failure mode from scratch.

This is the **third leg** of the TRT-LLM perf-knowledge tripod:

- `perf-optimization-casebook` — how to make things *faster*.
- `perf-regression-cookbook` — how things *got slower* (known-good → bad).
- `perf-instability-cookbook` (this skill) — how a metric became *variable*
  when the mean was fine (or the mean measurement itself was unreliable).

Every case is anchored to fix commit(s) + PR(s) — and to NVBug ID(s) when the
fix cites one — so any case can be deep-dived later.

This skill is **reference material, not a coordinator.** It does not run
profiling or edit code. It is consulted by `perf-analysis` when triaging a run
whose spread across reps / iters / ranks is anomalous, by reviewers assessing
the instability risk of a change (does it add mid-run JIT? does it manage a
collective resource via GC? does it opportunistically pick a fast collective
that can silently fall back?), and by agents debugging perf CI bar flakes.

## What "instability" means in this cookbook

An instability is any perf pathology whose signature is **variance rather than
a shifted mean**:

- **First-iteration / mid-run spikes** driven by lazy compile, JIT, capture,
  or cache warm-up landing in served or measured iterations.
- **Cross-run variance** — the same test / same commit / same cluster gives
  different numbers each rep, spread that swamps any real signal.
- **Cross-rank hangs / stalls** driven by non-deterministic timing across
  ranks (GC-driven collective destruction, log-flush races, uneven warmup).
- **Silent slow-path fallbacks** from opportunistic fast collectives that
  behave differently when a rare condition fires (autotuner cache miss during
  capture, OOM during registration, library-mismatch load failure).
- **Non-determinism that poisons the perf signal** — GC-driven barrier hangs,
  autotuner tactic ties broken by non-deterministic 0.0 timings: the *measured
  quantity itself* becomes unreproducible. Parked here rather than in the
  regression cookbook because the observable is variance, not a step.
  **This clause is narrow, and it excludes plain output nondeterminism.** An
  earlier revision read it as covering "log_probs differ across runs" and
  admitted a case (PR #15125) whose only varying quantity was the returned
  numbers — while `data/pending.yaml` simultaneously excluded three sibling
  output-variance PRs (#14841, #14509, #14652) by name for exactly that reason.
  The corpus cannot have it both ways, and the pending.yaml side is the right
  one: the test is **"does a perf metric vary?"**, not "does something vary?".
  Wrong numbers are a correctness defect however non-deterministically they
  appear, and they belong to functional triage. Ask which quantity moves before
  filing.
- **Measurement instabilities** — the *product* is deterministic but the
  *measurement* is not (NFS log-flush races, missing test knobs that dilute
  spec-decode acceptance-rate spread).

Everything here is a mitigation, not a regression fix — the entry either
introduces a warmup / prewarm / determinism knob that didn't exist, or hardens
a fallback path that could silently take the slow lane. When the fix is a
straight *revert of a regression that increased variance*, the case still lives
here (variance is the observable) but its `patterns:` will cite the regression
pattern too.

## When to Consult

- A metric (throughput / TTFT / ITL / P99 / accepted-tokens-per-step) is
  unstable across reps of the *same* configuration, and you need candidate
  failure modes ranked by prior frequency.
- The first iteration (or first N iterations) of a run is dramatically slower
  than the steady state, and you want to know whether that class has already
  been mitigated for a similar model / kernel / warmup path.
- A run **hangs or produces different tokens** across otherwise-identical
  invocations, and you suspect GC / autotuner-tie / collective-negotiation
  non-determinism rather than an accuracy defect.
- You are reviewing or writing a change that adds JIT compilation, opportunistic
  collective negotiation, GC-managed collective resources, or a warmup path
  and want the known ways such changes have destabilized perf.
- A perf CI bar / QA sweep flake is under investigation — the observable is
  intermittence, not a monotonic drop.
- You just mitigated an instability and want to record it so the next triage
  starts warmer.

## How to Use — the loop

1. **Characterize the variance.** Which metric is unstable, on what
   model/hardware/phase, and along which axis (rep-to-rep, iter-to-iter,
   rank-to-rank, first-vs-steady-state)? If you have a profile, note the
   signature (JIT/compile spans landing after iter 0? a rank blocked on a
   collective while peers proceed? an autotuner cache-miss warning during
   capture?).
2. **Match.** Normalize your terms via `data/aliases.yaml`, match an
   **instability pattern** in `data/patterns.yaml` (symptom signals →
   pattern), then grep case frontmatter for the pattern id or canonical
   terms and open only the winning case file(s):

       grep -l "pattern-<id>" references/*/*.md
       grep -l "warmup-coverage-gap" references/*/*.md

3. **Localize.** Use the matched cases' **Detection signal** to confirm or
   refute the class on your setup (grep the log marker, check the knob, look
   for the profile signature) before proposing a fix. Prior instances'
   **How introduced** tells you which kinds of commits to suspect first when
   an instability is fresh in the commit range.
4. **Mitigate & verify.** Adapt the precedent's **Fix mechanism**; measure
   both the mean *and* the spread of the metric that was unstable. Numbers
   come from real measurement only.
5. **Record.** Add or update a case (see "Adding a New Case"); instabilities
   recur in classes, and the next triage should start warmer.

### Consultation trace (machine-parseable — always emit)

Every consultation MUST end with exactly ONE trace line in your visible reply
text, as soon as the Match step resolves:

    INSTABILITY-COOKBOOK: matched=<family>/<case-slug> confidence=<high|medium|low> adapted="<one-line adaptation>"

or, when no precedent fits:

    INSTABILITY-COOKBOOK: no-match query="<instability class / signals you searched for>"

`<case-slug>` is the case file name without `.md`. Emit the line even when the
consultation is a dead end: it is the join key for automated audit trails, and
the no-match rate is the cookbook's coverage metric for maintainers.

## Corpus & provenance

- Source: TensorRT-LLM `main`, commits whose subject or PR describes a
  perf-instability mitigation (warmup / JIT coverage, opportunistic-collective
  fallbacks, determinism knobs, perf-harness races), classified from their PR
  descriptions and diffs. Window: 2026-02 → 2026-07, extended to **2026-08** by
  a second, independent sweep of the trailing year's merged PRs (2026-08-12);
  three of that sweep's candidates are cases here, and the rest are recorded in
  `data/pending.yaml` or `data/removed.yaml` with their reason.
- Current corpus (2026-08-12): **17 confirmed cases** across 5 modules, from
  ~36 candidate PRs (~10 excluded as accuracy-only, functional-only, or pure
  optimizations rather than instability mitigations) plus the 2026-08 PR-sweep
  additions and `ignore-eos-with-spec-decoding`.
- **Half the corpus did not survive its own evidence audit: 13 of the original
  26 cases were removed or moved on 2026-08-12**, after re-reading each fix
  PR's description and diff for a metric that actually *varies*.
  That ratio is the single most useful fact in this file: a commit sweep over
  warmup / collective / determinism paths selects for *mechanisms that could*
  destabilize a metric, and roughly half of them turn out to have no varying
  metric anywhere in the fix PR. Ten were removed outright and three were moved
  to the regression cookbook; every removal left its mechanism prose behind in
  `data/patterns.yaml` or the module `index.md`, and the full list with
  per-case reasons is in `data/removed.yaml`.
  - **The fix PR describes a non-perf defect** (a crash, hang, memory or
    accuracy fix, or a test-isolation fix): PR #13078 / 5955765
    (`warmup-more-accurate-launch-params`), PR #14536 / 6191524
    (`mla-cached-kv-maybe-compiled-cat-warmup` — a recompile that tripped the
    hang detector), PR #10340 / 5820734 (`general-warmup-memory-pool` — a
    peak-memory fix), PR #13758 / 5805494
    (`warmup-token-cap-protect-autotuner` — an int32 overflow / IMA at the
    16384-token warmup shape, i.e. a crash), PR #12015 / 5923949
    (`nccl-library-load-stability` — a load-time segfault on NCCL version
    mismatch), and PR #10730 / 5680911
    (`single-process-mode-env-cache` — unit tests silently no-op'ing).
  - **Not perf-instability**: `nccl-symmetric-fallback-long-context` (PR #11870
    — a hang that "consistently fails around round 30-40"),
    `trtllmgen-fmha-densify-grid` (PR #15305 — a warmup-grid hole for one
    narrow seqlen range, a deterministic coverage miss, now a regression case),
    `visual-gen-warmup-cache-key` (the observable was a spurious "not warmed
    up" WARNING, the recompile only ever *potential*) and
    `beam-search-logprobs-nondeterminism` (PR #15125 — an *output*-correctness
    defect: wrong `log_probs` varying with batch drain order, no perf metric
    anywhere, and it had been filed under `pattern-metric-with-hidden-rng`
    despite containing no RNG).
  - **Perf regressions misfiled as instability**, moved to the regression
    cookbook: `warmup-token-cap-revert` (PR #14252), `trtllmgen-fmha-jit-warmup`
    (PR #14851), and `mla-chunked-prefill-maybe-compiled-cat-warmup`
    (PRs #11743 / #11744 — a `torch.compile` paid once per process on a branch
    the warmup never enters).
  Read the whole bullet as two rules. First: "the symptom looked intermittent"
  is not evidence of an instability bug — a JIT or compile stall that fires on
  every cold start is a deterministic regression that merely *presents* as
  variance in a trace, because it moves the mean and not the spread. Ask
  whether the cost is paid once per process (regression) or unpredictably
  (instability) before choosing a cookbook. Second: a spurious warning, a
  segfault, a hang and a wrong output are all "flaky-looking" and none of them
  is a perf instability; require a metric that *varies* across otherwise
  identical runs.
- Every case carries `commits:`, `success_prs:`, `failed_prs:` and `nvbugs:`
  (bare IDs, possibly empty) frontmatter — the complete provenance for that
  mitigation. Patterns aggregate provenance transitively through their
  `instances:` list.
- **`failed_prs:` is the half to read before proposing a mitigation.** It lists
  the attempts at the same mitigation that did *not* land, with the reason in the
  case's **Failed attempts** bullet. For instability the two recurring rejection
  reasons are both review anti-patterns: the attempt
  *diluted* the variance (raised a CV tolerance, inflated `run_count`, widened a
  timeout) or *removed the signal* (waived / deprecated the case off the perf
  list) instead of removing the variance. `success_prs:` is what merged; an
  attempt still open is in neither list, because a case means "a mitigation is
  on `main`". Both cookbooks record only instabilities with a merged fix PR — a
  confirmed instability with no landed mitigation is not recorded at all, since
  a case exists to be cited for its **Fix mechanism**.
- **13 of the 17 cases carry no `nvbugs:`** — the corpus was built from
  commits on `main`, and those mitigations' PRs are titled `[None]` or against
  a JIRA / GitHub issue. The 4 that carry one are
  `deepgemm-paged-mqa-logits-prewarm`, `force-num-accepted-tokens-in-spec-perf-test`,
  `gen-only-log-flush-sentinel` and `ignore-eos-with-spec-decoding`. Do **not**
  back-fill an ID from a PR-number coincidence — the same PR can be a *culprit*
  for one defect and a fix for another (#13505 is both), and culprit-for-fix is
  the dominant false positive in this kind of match.
- Verdicts are established from **public PR descriptions, commits, diffs and
  GitHub issues only**. `nvbugs:` IDs are pointers, not evidence.
- Commits whose PR description was too unclear to confirm as instability
  mitigations (as opposed to functional / accuracy fixes) but which touch
  warmup / collective / determinism paths are parked in `data/pending.yaml`
  — the standing to-classify pool.

### What counts as an "instability" here vs a regression

The two cookbooks partition perf-bug fixes by observable:

- If the observable is **"one number went down and stayed down"** (a step
  function in the mean), it belongs in `perf-regression-cookbook`.
- If the observable is **"the number varies across reps / iters / ranks"**
  (spread, spikes, hangs, or measurement noise) or **"the first N iters are
  slower than the steady state,"** it belongs here.

A fix can span both: a warmup gap can regress the mean (steady state N iters
slower than before) *and* inflate the variance (iter 3 spikes by 30 s). When
that happens, prefer this cookbook if the mitigation lands **new warmup / new
determinism knobs**, and cross-link the regression case if one exists.

## Modules — routing table

`references/` is indexed by **module**: the piece of TRT-LLM the fix landed
in. Open the module **index** nearest your instability, then drill into a
single case file from its case table. The canonical id list, with a prose
description of each module's code, is `data/modules.yaml` — identical in
perf-regression-cookbook, so a module id queries both corpora.

**Filing rule (and reading rule): the module is where the FIX landed, not
where the symptom showed.** A missing FMHA warmup grid is fixed by editing
the warmup enumeration → `jit-and-warmup`, even though the spike appears in
an attention kernel. If you are unsure which module to open, use
`subsystems:` instead — it is multi-valued and crosses modules:
`grep -l 'attention-kernel' references/*/*.md`.

The 18-module vocabulary is shared; instability cases currently occupy five
of them.

| Module | Index | Covers | Cases |
|--------|-------|--------|-------|
| JIT & warmup | [references/jit-and-warmup/](references/jit-and-warmup/index.md) | first-iter / mid-run JIT stalls, warmup shape & mode coverage, warmup that only exists as another feature's side effect, cold-start ADP router imbalance | 7 |
| Communication | [references/communication/](references/communication/index.md) | opportunistic fast collectives (NCCL_SYMMETRIC, MNNVL) with silent fallbacks, in-graph registration failures, platform-conditional disables | 3 |
| MoE | [references/moe/](references/moe/index.md) | a2a backends whose finalizers fire collectives — non-deterministic GC across ranks | 1 |
| Scheduler & executor | [references/scheduler-and-executor/](references/scheduler-and-executor/index.md) | host-side GC / finalizer pauses landing in the decode loop because a tuning knob was never forwarded | 1 |
| Measurement & test | [references/measurement-and-test/](references/measurement-and-test/index.md) | metrics with hidden RNG (spec-decode acceptance), log-flush races between benchmark client and gen worker | 5 |
| Case schema + template | [references/case-template.md](references/case-template.md) | The field schema and rules for adding cases | — |

All five also hold **regression** cases in perf-regression-cookbook (which
additionally uses the other 13 modules) — check there when the same signature
also shows a one-way level shift rather than variance.

### Quick index — instability signature → where to look

Match the signature, open the pattern in `data/patterns.yaml`, then its
instance cases. Patterns are the **cross-module** axis: a pattern listing
several modules says this mechanism already recurred in code with nothing
else in common.

| Observed signature | Pattern | Modules |
|---|---|---|
| First iter (or iter 3) inflates P99 by seconds; steady state fine | `pattern-jit-on-hot-path` | jit-and-warmup |
| Warmup grid covered "typical" shapes but a live request lands on an uncovered shape | `pattern-warmup-coverage-hole` | jit-and-warmup |
| Warmup ran once, but the served *path* (image=None vs image=…, cached-kv vs no-cache) invalidates the compiled graph | `pattern-warmup-path-mismatch` | jit-and-warmup |
| First N requests all pinned to one worker until a shared prefix is cached everywhere; cold ranks stay cold | `pattern-cold-start-worker-imbalance` | jit-and-warmup |
| An opportunistic fast collective (NCCL_SYMMETRIC, MNNVL, ...) silently falls back to slow-path on OOM, cache miss during capture, or library mismatch | `pattern-opportunistic-collective-fallback` | communication |
| Platform / arch combination hits a NCCL feature bug — need a targeted disable rather than blanket off | `pattern-platform-conditional-disable` | communication |
| Ranks hang inside `__del__` / `atexit` collectives when Python GC fires at different times across ranks | `pattern-gc-driven-collective-destruction` | moe |
| Decode gaps with the device idle and the host in `gc.collect`; a GC tuning knob defaulted to None and was never forwarded | `pattern-gc-pause-on-decode-path` | scheduler-and-executor |
| Spec-decode perf number moves 5–20% between reps because acceptance rate is a per-run RNG draw | `pattern-metric-with-hidden-rng` | measurement-and-test |
| Benchmark client returns while gen-worker log is still flushing across NFS → parser misses lines | `pattern-cross-node-log-flush-race` | measurement-and-test |

## Case Schema

See `references/case-template.md`. Core fields: **Provenance** (nvbug when the
fix cites one · commit · PR — mandatory commit + PR), **Symptom (variance
signature)**, **Root cause**, **How introduced**, **Fix mechanism**,
**Detection signal**, **Prevention/guard**, **Generalizes to**.

## Principles

1. **Variance is the observable.** Every case describes a *spread* pathology
   — first-iter spike, cross-rank hang, rep-to-rep variance, measurement
   noise. If the observable is a mean shift with no spread growth, it belongs
   in the regression cookbook.
2. **Trace, don't recall.** Every number, knob, and path must be traceable to
   the fix PR's description or diff. Instability magnitudes appear only when
   the PR states them, with the source cited.
3. **Never present a precedent as a diagnosis.** A matched case is a
   hypothesis to *check* (via its Detection signal), not a conclusion.
4. **Patterns are the match surface.** Cases are instances; match on the
   pattern and the canonical signals, not on titles.
5. **Pending is not precedent.** `data/pending.yaml` entries are unconfirmed;
   promote them to cases only after their nature is established.
6. **Interface facts are as-of their pinned commit.** Knob names, env vars,
   and paths in a case describe the code at the case's `commits:`; re-verify
   against your checkout before acting.

## Adding a New Case

1. Confirm the fix is an instability mitigation (variance/spike/hang/race is
   the stated or clearly implied observable — not a monotonic slowdown).
2. Fold vs. new: a PR series mitigating the *same* instability (same root
   cause or same NCCL feature) is ONE case with `related:` provenance lines.
   A new case needs a new root cause or a new pattern.
3. Pick the module by **where the fix landed** (`data/modules.yaml`); create
   `references/<module>/<slug>.md` from `references/case-template.md`; fill
   every frontmatter field with canonical terms from `data/tags.yaml` —
   `module:` must equal the parent directory name; register/extend the pattern
   in `data/patterns.yaml` (bidirectional: case `patterns:` ↔ pattern
   `instances:`, and add the module to the pattern's `modules:` list); add a
   row to that module's `index.md` case table. A module id with no cases yet
   in this cookbook is still valid — create the directory and its `index.md`.
4. Keep the anti-fabrication rules of the template; if the evidence is too
   thin, park it in `data/pending.yaml` instead.
5. This cookbook ships in the public TensorRT-LLM repository. Cite NVBugs by ID
   only; every other fact (symptom numbers, bisect, root cause, verification)
   must come from a public PR description, commit, diff, or GitHub issue. Never
   copy or paraphrase NVBug descriptions, comments, titles, status, severity,
   or people.

## Relationship to Other Skills

- **perf-regression-cookbook** — the mean-shift twin. When an instability
  fix is really the revert of a regression that inflated variance, cross-link
  the regression case.
- **perf-optimization-casebook** — the offensive twin. Some instabilities
  are optimizations that traded worst-case for best-case; cross-link when a
  casebook entry maps to the same mechanism.
- **perf-analysis / perf-host-analysis / perf-nsight-systems** — classify
  and localize the variance signature; this cookbook then maps signatures to
  failure modes.
- **trtllm-code-contribution** — the Prevention/guard field feeds review
  checklists for changes that add JIT / opportunistic collectives / GC-managed
  collective resources.
