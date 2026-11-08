---
name: perf-optimization-casebook
description: >
  Casebook of past successful and classic TensorRT-LLM optimizations
  (runtime/execution and kernel level) recorded as reusable decision
  precedents. Consult when deciding
  which optimization to apply for a classified bottleneck or a given
  config/model/hardware, to find prior art and adapt a proven approach
  instead of guessing. Each case records applicability signals, mechanism,
  how to apply, expected effect, accuracy risk, verification, and rollback.
tags:
  - optimization
  - casebook
  - precedents
  - decision-support
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Performance Optimization Casebook

A curated library of **classic and previously-successful TensorRT-LLM
optimizations**, written as decision precedents. Use it to answer "for *this*
bottleneck / config / model / hardware, what has worked before, and how do I
adapt it?" — instead of inventing an approach from scratch.

This skill is **reference material, not a coordinator.** It does not run
profiling, edit code, or route to specialists by itself. It is consulted *by*
the coordinators (`perf-analysis` after a bottleneck is classified,
`perf-optimization` before prioritizing/routing) and by the `perf-sweep`
system when choosing what to try next. Implementation always goes through the
relevant specialist/skill named in each case.

## When to Consult

- A bottleneck has been classified (compute / memory / launch / communication /
  sync) and you need candidate optimizations for that class.
- You are about to prioritize or route an optimization and want to check for
  prior art, known pitfalls, and the right specialist.
- You are choosing the next config to try in a serving sweep.
- You just landed a verified win and want to record it for future reuse.

## How to Use a Case — the loop

1. **Classify.** Know the current bottleneck class and the workload context
   (model arch, hardware/SM, ISL/OSL, concurrency, current config). If you do
   not have this, get it from `perf-analysis` / profiling first.
2. **Match.** Search wide with free text, then rank — never let the query
   vocabulary narrow what is reachable:
   - **Recall (grep the whole case files, free text):** grep the case bodies
     with your own natural query terms — model name, knob, mechanism word,
     deployment feature — over the full files, not just structured fields:

         grep -ril "lora" references/                 # any term, whole file
         grep -ril "chunked prefill" references/

     Every concept a case covers lives in its prose (**Applies when** /
     **Generalizes to** / mechanism), so a free-text grep over the whole file
     is recall-complete on its own. Frontmatter is **additive** structure layered
     on top — it never removes a word grep could already find. Do **not** pre-
     translate the query into canonical vocabulary before searching; that is
     what would make grep miss LoRA / beam search / a reasoning model's thinking
     phase and other concepts that have no canonical signal. Browse the family
     **index** too if you want a curated table, but grep alone does not miss.
   - **Rank (optional accelerator):** among the hits, optionally use the
     frontmatter fields (canonical `signals`/pattern grep — see "Frontmatter
     fields" below) to decide which case file(s) to open first. This only
     *reorders* the free-text hits — it never defines or bounds them.
   - Then open **only** the winning case file(s). Match on signals, not on the
     optimization name.
3. **Adapt — do not copy.** A case is a precedent, not a guaranteed config.
   Adjust knob values to the current model/hardware; read the nearest
   checked-in config and the model's deployment guide before setting numbers.
4. **Verify.** Apply through the named specialist/skill, then measure. Numbers
   come from real measurement only (see Principles). A precedent's expected
   *direction* of effect does not excuse skipping measurement.
5. **Record.** If the change is a verified win (or a clear, instructive
   failure), add or update a case so the next decision is better informed (see
   "Adding a New Case").

### Frontmatter fields (optional precision — not a search step)

Cases carry YAML frontmatter (vocabulary: `data/tags.yaml`; spec:
`references/case-template.md`). It is **additive structure for ranking and
judging** the free-text hits from Match step 2 — never a second search you must
run, and never a filter that bounds what grep already found. Use it as needed:
grep a canonical `signals`/pattern id (`data/patterns.yaml`) to rank hits, and
read a candidate's `eligibility:` / `interactions:` before recommending (respect
`incompatible-with`; follow `depends-on` to the producer case first). A synonym
won't hit these fields (they store canonical terms only) — that is fine, because
the free-text grep already found the case; never emit `no-match` from an empty
field grep.

Legacy (not-yet-migrated) cases have no frontmatter — for those, fall back
to the family index tables. When adding or editing a migrated case, keep the
contract by hand: frontmatter values must be canonical terms from
`data/tags.yaml` (never synonyms), and case↔pattern pointers must stay
bidirectionally consistent (`patterns:` in the case ↔ `instances:` in
`data/patterns.yaml`).

### Consultation trace (machine-parseable — always emit)

Every consultation MUST end with exactly ONE trace line in your visible reply
text, as soon as the Match step resolves:

    CASEBOOK: matched=<family>/<case-slug> confidence=<high|medium|low> adapted="<one-line adaptation>"

or, when no precedent fits:

    CASEBOOK: no-match query="<bottleneck class / signals you searched for>"

`<case-slug>` is the case file name without `.md`. Emit the line even when the
consultation is a dead end: it is the join key for automated audit trails
(which precedents were consulted vs cited vs hallucinated), and the no-match
rate is the casebook's coverage metric for maintainers. One line per consulted
question; never summarize multiple matches into one line. A `no-match` line
asserts your free-text recall grep over the whole case files (Match step 2)
turned up nothing relevant — it must never be emitted merely because a
canonical-field grep returned empty.

## Case Families — routing table

Open the family **index** relevant to your classified situation, then drill into a
single case file from its case table (each family is a directory of one-file-per-case
behind an `index.md` of patterns + a case-picker table).

| Family | Index | Covers |
|--------|-------|--------|
| Runtime / execution | [references/runtime-execution/](references/runtime-execution/index.md) | Overlap scheduler, PDL, CUDA graphs (piecewise/padding, split-op-for-capture), multi-stream overlap, MLA KV-cache reuse & activation freeing, auxiliary-cache management, online-EPLB overlap, chunked prefill (aligned auxiliary), sparse-attention path selection, host-overhead & sync-free, batching/scheduler knobs, speculative decoding / MTP |
| Kernel & fusion | [references/kernel-and-fusion/](references/kernel-and-fusion/index.md) | Op/epilogue fusion (AR+residual+norm+quant, add+norm+quant, QK-norm+RoPE, pre-pass folding, data-movement+quant), fused attention & MLA kernels, sparse-MLA top-k attention & top-k selection kernels, MoE grouped-GEMM backends (trtllm-gen / DeepGEMM / MegaMoE), low-precision GEMM (FP8 block-scale / W4A8 / NVFP4), ranking-only precision, routing kernels, custom Triton/CuTe/TileIR, GEMM tactics |
| Communication | [references/communication/](references/communication/index.md) | EP all-to-all (DeepEP, low-precision dispatch/combine), TP AllReduce (MNNVL two-shot kernel, UserBuffers / symmetric memory, shape-aware autotune), collective kernel & strategy selection |
| Case schema + template | [references/case-template.md](references/case-template.md) | The field schema and a worked example for adding cases |

Many runtime/kernel/communication cases were distilled from the TRT-LLM
DeepSeek-R1/V3 serving stack (MLA attention, fine-grained MoE, MTP, attention-DP +
expert-parallel), where these levers cluster — but each is recorded as a
**transferable pattern** (see *Generalizes to*); match on signals, not on the
DeepSeek origin. Each carries a `Commits` line so you can `git show` the source.

### Quick index — bottleneck signal → where to look

Routing shortcut once the bottleneck is classified. Match the signal, open the
family index, then match a case's **Applies when** / **Generalizes to** in its
case table and open that case file.

| Profile signal | Bottleneck | Family → candidate patterns |
|---|---|---|
| GPU idle between steps; host prep on critical path | launch / host | runtime → overlap scheduler · move per-step bookkeeping into C++/fused op · piecewise CUDA graph · CUDA-graph padding |
| Many small **dependent** kernels, SM≥90 | launch | runtime → PDL · kernel → mega-fuse MoE / fuse collective epilogue |
| Residual + Norm (+quant) chain of small kernels | memory / launch | kernel → fuse AR epilogue · fuse local add+norm+quant · QK-Norm+RoPE |
| AllReduce / all-to-all a large share at TP/EP>1 | communication | communication → DeepEP · low-precision dispatch/combine · MNNVL two-shot kernel · UserBuffers · shape-aware autotune |
| Low MFU under data-parallel with ragged load | host / compute | runtime → eliminate attention-DP padding · CUDA-graph padding |
| KV reads dominate decode (long ctx / high concurrency) | memory | kernel → FP8 MLA KV · runtime → MLA KV-cache reuse · free MLA intermediates |
| Sparse-attention model (DeepSeek V3.2 / DSA): full-seq attention wasteful; lightning-indexer + top-k on the hot path | compute / launch / memory | kernel → sparse-MLA top-k attention · specialize top-k selection kernel · ranking-only TF32 · fuse data-movement+quantize · runtime → skip sparse when short-seq degenerate · chunked prefill (aligned auxiliary) · auxiliary cache in KV manager · split custom op for piecewise capture |
| MoE grouped-GEMM dominates the step | compute | kernel → trtllm-gen FP4 MoE · mega-fuse MoE · HW-matched low-precision GEMM · optimize routing kernel |
| Expert load imbalanced across EP ranks | comm / sync | runtime → online-EPLB + overlap rebalance |
| Spec-decode accept rate low / draft forward exposed | compute / launch | runtime → relaxed acceptance · two-model MTP-Eagle overlap |
| OOM caps batch / KV budget | memory | runtime → free MLA intermediates · kernel → low-precision GEMM/KV |

## Case Schema

Every case records these fields (see `references/case-template.md` for the
canonical form):

- **Commits** *(optional)* — primary `<hash> subject (#PR)` plus folded-in
  `related:` follow-ups, when the case is distilled from specific commits, so a
  reader can `git show` the source. Omit when there is no upstream commit.
- **Applies when** — bottleneck class + signals (model arch, hardware/SM,
  workload shape, current config) that make the case relevant.
- **Counter-signals** — when NOT to apply: auto-disable regimes, degenerate
  shapes, negligible-share workloads, unmet requirements. Check this before
  recommending a matched case.
- **Mechanism** — why it helps, in one or two lines.
- **Generalizes to** — the transferable *pattern* this case instantiates and the
  adjacent situations it carries to (other model/arch, neighbouring op-chains,
  different knob values), plus what to change when adapting. This is what an
  agent matches on when it does **not** hit the exact instance — the casebook's
  main job. Match on this pattern, not the case title.
- **Apply via** — the exact config knob and/or the specialist/skill to delegate
  implementation to. Never a hand-written runnable config invented here.
- **Expected effect** — qualitative direction and the metric that should move.
  Quantitative deltas only when measured (cite the run) — see Principles.
- **Accuracy risk** — `lossless`, `lossy`, or `mixed` (a case bundling moves
  of differing risk — it must state which part is lossy). Lossy cases require an on-disk
  accuracy record plus a rollback criterion before being promoted as a "best"
  config (mirrors the perf-sweep correctness gate).
- **Verify** — what to measure to confirm the win and (for lossy) parity.
- **Rollback** — how to revert and the regression trigger that mandates it.
- **Prior art** — pointers to the owning skill, TRT-LLM repo docs, or PRs.

## Principles

1. **Adapt, don't copy.** Cases are precedents. Knob values, hardware
   constraints, and model arch differ — re-derive numbers from the nearest
   checked-in config and the model guide. Treat any value carried over from a
   case as unverified until measured here.
2. **Never fabricate numbers.** Quantitative before/after figures must come
   from real profiling/benchmark output, with the source cited. A case states
   the *direction* of effect and the *metric to watch*; it does not assert a
   speedup that was not measured. Do not present casebook expectations as
   achieved results.
3. **Respect the accuracy gate.** Optimizations split into:
   - **Lossless** (output-equivalent): scheduling/batching/graph knobs, and
     backend *selection* (`attn_backend`, `moe_config.backend` — same math,
     different implementation). Safe to evaluate on throughput/latency alone.
   - **Lossy** (output-affecting): KV-cache quant (`kv_cache_dtype` fp8/int8/
     nvfp4), weight/GEMM quant (`quant_algo`), low-precision MoE combine,
     low-precision EP dispatch/combine, **relaxed spec-decode acceptance** — and
     **PDL** (`TRTLLM_ENABLE_PDL`), which changes inter-kernel ordering/visibility
     and has produced real NaN/accuracy regressions per kernel (verify per
     kernel). Spec/MTP is output-equivalent *by design* but can drift on
     acceptance/draft bugs, so verify on first enable. Lossy wins need an
     accuracy record + rollback criterion, never promoted as "best" without it.
   This split matches `perf-sweep-workflow` `core/gate.py` — keep them aligned.
4. **One change at a time, measure each.** Combined changes hide regressions
   and confound attribution. Roll back on >5% slowdown or any accuracy
   regression outside tolerance.
5. **Interface facts are as-of their pinned commit.** Knob names, config
   bounds, file paths, and function names in a case describe the code at the
   case's `commits:` — they are historical anchors, not present-tense truths.
   Before acting on them (especially machine-evaluable `eligibility:`
   entries), re-verify against YOUR TRT-LLM checkout; cases point to where
   the live value lives (e.g. a `file::symbol` guard). A mismatch means the
   code moved since the pin — prefer the live source. Mechanism and pattern
   statements ("why it helps") are not version-bound; only interface facts
   are.

## Adding a New Case

After a verified win (or an instructive failure):

1. Pick the family directory (`references/<family>/`); if none fits, propose a new
   family dir + `index.md` and add a row to the routing table above.
2. Create `references/<family>/<slug>.md` from the schema in
   `references/case-template.md` — H1 title + the breadcrumb line + every field
   bullet — then add one row to that family index's case table (Case → file,
   Applies-when, Generalizes-to, Risk), and, if it instantiates a listed pattern,
   link it from that pattern's `_(Instance: …)_`. Put real measured deltas with the
   source run; otherwise write "measured Δ to be recorded from <run>".
3. Keep it a *decision precedent* — applicability, mechanism, knob/specialist,
   risk, verification, rollback. Fill **Generalizes to** with the pattern the
   case instantiates and 2–4 adjacent situations it carries to: this is what
   lets an agent reuse the case when its situation is a *variation*, not the
   exact instance (match on the pattern, not the title). Do **not** re-explain
   subsystem architecture (how MoE/attention/KV cache work); link to the TRT-LLM
   repo developer guides for that (see
   `docs/design/agent_toolkit_organization.md` §2.3).
4. Cross-link the owning specialist/skill so implementation stays delegated.

### Ingestion discipline (agent-scanned corpus)

This corpus is scanned and written by agents. Selection — "is this a
transferable optimization precedent?" — stays with the writing agent's
judgment; the rules below target known agent failure modes (fabricated
detail, plausible over-generalization, cross-run divergence), not judgment.

- **Trace, don't recall.** Every number in a case (bounds, caps, shapes,
  dims) must be traceable to a specific line in a cited commit's diff or
  the pinned source; every model/arch applicability claim must be checked
  against the modeling source (e.g. the routing-method enum), never
  inferred from model-family resemblance. If you didn't look it up, don't
  write it.
- **Fold vs. new.** A follow-up PR that iterates the *same decision* (same
  pattern, same knob/kernel surface) folds into the existing case as a
  `related:` commit — a PR series refining one optimization is ONE case.
  A new case requires new applicability signals or a new pattern.
- **Stub threshold.** If the four core slots (applicability signals,
  mechanism, transferable pattern, apply-via) can be at least roughly
  filled, record the case with `maturity: stub` so it enters the match
  surface. If not even roughly — skip it entirely; do not park
  half-guesses in the corpus.
- **Verify before landing.** A case is added or upgraded only after an
  independent fact-check pass against its cited commits (quantitative
  claims, knob names, file paths, model-applicability claims). The writing
  agent's recall is not evidence.
- **Scan cursor.** Sweeps of TRT-LLM main record the last commit scanned
  here and resume from it. Deduplicate against the corpus by grepping the
  hash across case `commits:`/`Commits:` fields before distilling.
  Current cursor: *not yet recorded — the pre-pilot corpus was swept ad
  hoc; set this at the next sweep.*

## Relationship to Other Skills

- **perf-analysis** classifies the bottleneck → this casebook supplies
  candidate optimizations for that class.
- **perf-optimization** prioritizes and routes → consult this casebook for
  prior art and the right specialist before delegating.
- **perf-sweep-workflow** chooses configs to try → cases inform hypotheses;
  the deterministic core still validates and the challenger still gates.
- **Implementation** is always delegated to the specialist/skill named in the
  case (kernel-*, perf-torch-cuda-graphs, perf-torch-sync-free,
  perf-host-optimization, trtllm-serve-config-guide, etc.).
