# Case Schema & Template

Every casebook entry follows this schema. Copy the template below and fill
**every** field. A case is a *decision precedent* — applicability, mechanism,
how to apply, expected effect, accuracy risk, verification, rollback — not a
re-explanation of how a subsystem works and not a guaranteed config.

## Machine-readable frontmatter (pilot)

New and migrated cases carry a YAML frontmatter block **above the H1**. It
replaces the free-text `Tags:` bullet as the match surface: every value is a
canonical term from `data/tags.yaml` (synonyms belong in `data/aliases.yaml`,
never in frontmatter), and `patterns:` points into the registry
`data/patterns.yaml` — split out **each** transferable pattern the case
instantiates (a case often carries 2–3; each is its own match entry point).
Keep three invariants by hand when adding or editing a case: vocab
membership (canonical terms only), `id`/`family` structural identity
(`id` = `case-` + filename stem; `family` = parent directory), and
case↔pattern bidirectional consistency with `data/patterns.yaml`. Skeleton:

```yaml
---
id: case-<file-slug>              # must equal "case-" + filename stem
type: case
family: <dir name>                # runtime-execution | kernel-and-fusion | communication
maturity: stub | full | verified-locally
bottleneck: [<tags.yaml::bottleneck>]     # 1–3
signals: [<tags.yaml::signals>]           # 2–5 observable profile signals
architectures: [<tags.yaml::architectures>]
model_scope: [<tags.yaml::model_scope>]   # include model-agnostic if it transfers
phase: [prefill | decode | any-phase]
patterns: [<data/patterns.yaml ids>]      # EVERY pattern the case instantiates
accuracy_risk: lossless | lossy | mixed
apply_via_kind: [config-knob | env-var | default-on | kernel-change | code-change]
knobs: []                         # exact knob/env names, if any
specialists: []                   # owning specialist/skill names
commits: []                       # 10–12 char hashes, quoted (YAML eats digits)
log_markers: []                   # verbatim strings to grep in server logs
                                  #   (e.g. a fallback warning) — log evidence
eligibility: []                   # machine-checkable applicability constraints
                                  #   an agent can evaluate against the config.
                                  #   Volatile bounds: record a source pointer
                                  #   (file::symbol of the live guard) PLUS the
                                  #   values "as of <commit>" — never a bare
                                  #   number with no anchor (it reads as
                                  #   present-tense truth and rots silently)
interactions: []                  # typed relations; each entry:
                                  #   {case: case-<slug> | feature: <name>,
                                  #    relation: <tags.yaml::relations>, note: "..."}
                                  #   quote notes containing commas (YAML flow)
measured: []                      # verified-result records, appended at Record
                                  #   time; each entry requires ALL six fields:
                                  #   {metric, value, gpu, workload, run_ref, date}
                                  #   — no partial records (anti-fabrication)
---
```

The human-readable bullets below remain the case body; frontmatter carries
the *matching* metadata, bullets carry the *reasoning*. Legacy cases still
use the `Tags:` bullet until migrated.

## Field definitions

- **Tags** — *(legacy — replaced by frontmatter on migrated cases.)*
  At-a-glance match line: family, bottleneck class(es), and
  model/hardware applicability, comma-separated. Terse; it aids scanning while
  the bullets below carry the detail.
- **Commits** — *Optional provenance.* When the case is distilled from specific
  TRT-LLM commits, list the primary `<12-char hash> <subject> (#PR)` and any
  `related:` follow-ups (the iterations folded into this one case). Lets a reader
  `git show <hash>` the exact change. Omit for cases recorded from a local win
  with no upstream commit yet; `Prior art` then carries the pointers.
- **Applies when** — The bottleneck class (compute / memory / launch /
  communication / sync) plus the signals that make this case relevant: model
  architecture (dense / MoE, GQA / MLA), hardware/SM, workload shape (ISL/OSL,
  concurrency), and the current config state. Match on these signals, not on
  the optimization's name.
- **Counter-signals** — When NOT to apply: regimes where the mechanism
  auto-disables, degenerates, or actively hurts; workload shapes where the
  targeted cost is a negligible share; requirements the case cannot meet
  (e.g. bit-exact tie-breaking). Negative knowledge prevents misapplication —
  an agent matching only positive signals will over-apply.
- **Mechanism** — Why it helps, in one or two lines. When the case bundles
  several numbered sub-moves (each with its own commit), keep **Accuracy
  risk / Verify / Rollback decomposed by the same numbering** so a reader
  knows which move to revert when one is implicated.
- **Generalizes to** — The transferable *pattern* this case instantiates and the
  adjacent situations it carries to: other model/architecture, neighbouring
  op-chains, different knob values — plus what must change when adapting. An
  agent rarely hits the exact instance; this field is how it recognises that its
  situation is a *variation* of a known precedent. Name the pattern, list 2–4
  transfer targets, and state the adapt delta. Match on this, not the title.
- **Apply via** — The exact config knob(s) and/or the specialist/skill that
  owns the implementation. Never an invented runnable config; delegate.
- **Expected effect** — The qualitative direction of the effect and the metric
  that should move. Put a quantitative delta **only** if it was measured, and
  cite the run. Otherwise write "measured Δ to be recorded from <run>".
- **Accuracy risk** — `lossless` (output-equivalent), `lossy`
  (output-affecting), or `mixed` (a case bundling moves of differing risk —
  state explicitly which part is lossy). Lossy cases/parts require an on-disk
  accuracy record plus a rollback criterion before promotion as a "best"
  config. This mirrors the `perf-sweep-workflow` correctness gate
  (`core/gate.py`).
- **Verify** — What to measure to confirm the win; for lossy cases, also the
  parity/accuracy check. Include at least one **executable pointer** in
  backticks — a test path, a profiling command, or the knob to A/B — not only
  a described observation.
- **Rollback** — How to revert and the regression trigger that mandates it
  (default: >5% slowdown or any accuracy regression outside tolerance).
- **Prior art** — Pointers to the owning skill, TRT-LLM repo developer guides,
  or PRs. Keep links, not copied architecture text.

## Template

Each case is its **own file** at `references/<family>/<slug>.md`: an H1 title
(short and descriptive), a one-line breadcrumb back to the family index and this
schema, then every field as a bold bullet. Copy this raw form:

```
---
<frontmatter — copy the skeleton from "Machine-readable frontmatter (pilot)"
above and fill every field with canonical terms from data/tags.yaml>
---

# <short descriptive title>

> Part of the [<Family> casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** *(optional)* `<12-char hash>` <subject> (#PR); related:
  `<hash>` <subject> (#PR).
- **Applies when:** <bottleneck class> + <signals: model arch, SM, ISL/OSL,
  concurrency, current config>.
- **Counter-signals:** <when NOT to apply: auto-disable regimes, degenerate
  shapes, negligible-share workloads, unmet requirements>.
- **Mechanism:** <why it helps>.
- **Generalizes to:** <the pattern this instantiates>; carries to <2–4 adjacent
  targets: other arch / op-chains / knob values>; adapt by <what changes>.
- **Apply via:** `<config knob>` and/or delegate to **<specialist/skill>**.
- **Expected effect:** <direction> in <metric>. Measured Δ to be recorded
  from <run>, or: <metric> X → Y (source: <run/PR/doc>).
- **Accuracy risk:** lossless | lossy | mixed (if lossy/mixed: requires
  accuracy record + rollback criterion before promotion).
- **Verify:** <metric(s) to measure>; <parity check if lossy>.
- **Rollback:** <how to revert>; trigger: <regression condition>.
- **Prior art:** <skill / repo doc / PR pointers>.
```

## Worked example

A complete filled-in case, in the exact form it takes in its own
`references/<family>/<slug>.md` file — H1 title, breadcrumb, then every field:

```
---
id: case-cap-cuda-graph-batch-sizes   # = "case-" + filename stem
type: case
family: runtime-execution
maturity: full
bottleneck: [launch, memory]
signals: [gpu-idle-between-steps, many-small-kernels, small-batch-decode, memory-capacity-bound]
architectures: [any-sm]
model_scope: [model-agnostic]
phase: [decode]
patterns: []          # register the pattern in data/patterns.yaml first, then point here
accuracy_risk: lossless
apply_via_kind: [config-knob]
knobs: [cuda_graph_config.max_batch_size, cuda_graph_config.batch_sizes]
specialists: [perf-torch-cuda-graphs]
commits: []
measured: []
---

# Cap CUDA-graph capture to reachable decode batch sizes

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Applies when:** launch-overhead-bound decode (many small kernels, high host
  time, GPU idle between steps) **and** graph workspace memory is tight — e.g.
  a low-concurrency deployment where the server `max_batch_size` is large but
  only small decode batches are ever reached.
- **Counter-signals:** concurrency that routinely reaches batch sizes above
  the cap (those fall back to eager and regress step time); ample free memory
  (capping buys nothing — leave the default, which follows `max_batch_size`).
- **Mechanism:** CUDA graphs replay a captured launch sequence, removing
  per-step launch overhead. Capturing graphs for batch sizes that never occur
  wastes graph workspace memory; capping capture to reachable sizes keeps the
  launch-overhead win without the memory cost.
- **Generalizes to:** the pattern "size a captured/pre-allocated resource to the
  *reachable* working set, not the configured maximum." Carries to other
  resources keyed off `max_batch_size` — padded attention/decode buffers,
  speculative draft-length capture, warmup shape sets. Adapt by re-deriving the
  reachable set from the actual concurrency/ISL/OSL, then capping to it.
- **Apply via:** `cuda_graph_config.max_batch_size` / `cuda_graph_config.batch_sizes`;
  delegate capture/replay work to **perf-torch-cuda-graphs** /
  **perf-torch-cuda-graph-specialist**. (Default `cuda_graph_config.max_batch_size`
  to the server `max_batch_size`; lower it only when memory is the constraint.)
- **Expected effect:** lower per-step latency / smaller inter-step GPU-idle gap
  at the same accuracy; reclaimed graph workspace memory. Measured Δ to be
  recorded from the trial run dir.
- **Accuracy risk:** lossless — CUDA graph replay does not change numerics.
- **Verify:** per-decode-step time and GPU-idle ratio from an nsys trace before
  vs after; confirm batches above the cap fall back to eager (slower, not
  erroring) and are not on the hot path.
- **Rollback:** remove or raise `cuda_graph_config.max_batch_size`; trigger:
  any reachable batch size falling out of capture and regressing step time
  >5%, or graph capture OOM.
- **Prior art:** `perf-torch-cuda-graphs` skill;
  `trtllm-serve-config-guide` `references/knob-heuristics.md` (the DeepSeek-R1
  conc=1 example uses `cuda_graph_config.max_batch_size: 1` with server
  `max_batch_size: 512`).
```
