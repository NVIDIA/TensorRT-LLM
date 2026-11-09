# Instability-Case Schema & Template

Every cookbook entry follows this schema. An instability case is a *diagnostic
precedent* — what varied, how the variance was detected, why the mode / path /
collective / measurement was unstable, how it was mitigated, and how to catch
the same class earlier next time. It is **not** an optimization recipe (that
is `perf-optimization-casebook`'s job) and **not** a regression case (that is
`perf-regression-cookbook`'s job — the observable there is a mean shift, here
it is a variance / spike / hang).

Cases are distilled from perf-instability fix commits on TensorRT-LLM `main`.
Every case must be traceable: commit hash(es) and PR number(s) are mandatory
provenance; NVBug ID(s) are recorded, as bare IDs, when the fix PR or commit
names one. Every fact in a case comes from a public source only: the PR
description, the commit message, the diff, or a GitHub issue. An NVBug ID is a
pointer, never a source — do not copy or paraphrase NVBug descriptions,
comments, titles, status, severity, or people.

## Machine-readable frontmatter

Every case carries a YAML frontmatter block **above the H1**. Every value is a
canonical term from `data/tags.yaml` (synonyms belong in `data/aliases.yaml`,
never in frontmatter), and `patterns:` points into the registry
`data/patterns.yaml`. Keep three invariants by hand when adding or editing a
case: vocab membership (canonical terms only), `id`/`module` structural
identity (`id` = `case-` + filename stem; `module` = parent directory name),
and case↔pattern bidirectional consistency with `data/patterns.yaml`. Skeleton:

```yaml
---
id: case-<file-slug>              # must equal "case-" + filename stem
type: instability-case
family: <tags.yaml::family>       # coarse how-it-broke LABEL only;
                                  # it does not name a directory
module: <dir name>                # data/modules.yaml; MUST equal this
                                  # file's parent directory name. Pick by
                                  # where the FIX landed, not where the
                                  # symptom showed.
maturity: stub | full             # full = distilled from PR description + diff;
                                  # stub = core slots roughly filled only
instability_class: [<tags.yaml::instability_class>]   # 1–2
signals: [<tags.yaml::signals>]   # 2–5 observable variance signals
subsystems: [<tags.yaml::subsystems>]
introduced_via: [<tags.yaml::introduced_via>]  # how the instability got in,
                                  # when the PR/diff makes it identifiable;
                                  # else [unknown]
phase: [prefill | decode | warmup | any-phase]
patterns: [<data/patterns.yaml ids>]  # EVERY pattern the case instantiates
nvbugs: ["<id>"]                  # ALL NVBug IDs this case covers (empty if none)
commits: ["<12-char hash>"]       # ALL fix commits on main, quoted
success_prs: [<PR numbers>]       # fix PRs that LANDED (merged on main). Every
                                  # hash in commits: comes from one of these.
failed_prs: [<PR numbers>]        # fix ATTEMPTS at the same mitigation that did
                                  # NOT land: closed unmerged, superseded, or
                                  # rejected on review. [] when there were none.
                                  # An attempt still OPEN is neither — leave it
                                  # out until it merges or is closed.
---
```

The human-readable bullets below remain the case body; frontmatter carries the
*matching* metadata plus the full provenance, bullets carry the *reasoning*.

## Field definitions (body bullets)

- **Provenance** — `commit <12-char hash> · PR #<n>` per fix, with the commit
  subject; include `nvbug <id>` (ID only) when the fix PR or commit names one. When an instability
  was fixed by a series, list the primary first and the follow-ups as
  `related:` lines (fold-vs-new rule: a PR series fixing the *same*
  instability is ONE case).
- **Failed attempts** — one line per PR in `failed_prs:`: `PR #<n> — <what it
  tried> · <why it did not land>`. Omit the bullet entirely when `failed_prs`
  is `[]`. This is the highest-value half of the provenance for an agent about
  to propose a mitigation: it is the record of which plausible fixes were
  *tried and rejected*, so the next attempt does not re-propose one. Two
  rejection reasons recur for instability specifically and are worth naming
  verbatim when they apply: the attempt **diluted** the variance instead of
  removing it (raised a CV tolerance, inflated `run_count`, widened a timeout —
  both review anti-patterns), or it **removed the signal** rather than
  the variance (deprecated/waived the case off the perf list). Give the reason
  from the PR's own review thread or closing comment, never a guess. An attempt
  that is still open belongs here only once it closes.
- **Symptom (variance signature)** — what varied, along which axis (rep-to-rep,
  iter-to-iter, rank-to-rank, first-vs-steady-state), on what
  model/config/hardware if the PR names them, and how it surfaced
  (perf CI flake, QA sweep, profile) as stated in the PR or issue.
- **Root cause** — the actual defect, in one or two lines, as established by
  the PR description/diff. Say what broke, not just where. Distinguish *why the
  variance appeared* from *why the mean was slow*.
- **How introduced** — the change that caused the instability (commit/PR if the
  fix PR identifies it; else the code motion described, e.g. "warmup grid was
  authored covering only cudagraph batch buckets"; else "unknown — not stated
  in the PR").
- **Fix mechanism** — what the fix does, in one or two lines. For instabilities
  this is almost always one of: add a warmup, densify a warmup grid, harden a
  fallback path, add an explicit release, add a determinism knob, remove a
  process-lifetime cache.
- **Detection signal** — how an agent would spot this class: the profile
  signature (nsys/ncu span reappearing on iter 3, cross-rank divergence),
  log line, config check, or metric-spread delta to look for. Include at least
  one **executable pointer** in backticks (a grep, a knob to inspect, a
  profile view) — not only a described observation.
- **Prevention/guard** — what would catch this class earlier: the assert,
  unit/perf test, log warning, or review checklist item. If the fix PR added
  such a guard, name it; else state the gap.
- **Generalizes to** — the transferable instability *pattern* this instantiates
  and 2–4 adjacent situations where the same failure mode can recur. Match on
  this, not the title.

## Anti-fabrication rules

- Every number, knob name, and file path must be traceable to the fix PR's
  description or diff. If you didn't look it up, don't write it.
- Quantitative variance magnitudes ("+30 s spike on iter 3", "CV drops from
  8% to 1.2%") appear ONLY when a public PR description, commit, diff, or
  GitHub issue states them; cite which one. A number that exists only in an
  NVBug is not recorded.
- Model/hardware applicability comes from the PR text or diff, never inferred
  from resemblance.
- If the PR description is too thin to fill Root cause + Fix mechanism at
  least roughly, the entry belongs in `data/pending.yaml`, not here.

## Template

Each case is its **own file** at `references/<module>/<slug>.md`: an H1 title
(short, states the instability, not the fix), a one-line breadcrumb back to
the module index and this schema, then every field as a bold bullet:

```
---
<frontmatter — copy the skeleton above; canonical terms from data/tags.yaml>
---

# <short title naming the instability>

> Part of the [<Module display name> instability cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** commit `<12-char hash>` · PR #<n> — <subject>;
  related: commit `<hash>` · PR #<n> — <subject>.
- **Failed attempts:** PR #<n> — <what it tried> · <why it did not land>.
  (omit this bullet when `failed_prs: []`)
- **Symptom (variance signature):** <what varied along which axis> on
  <model/config/HW if stated>; surfaced via <perf CI / QA / profile>.
- **Root cause:** <the defect>.
- **How introduced:** <causing change, or "unknown — not stated in the PR">.
- **Fix mechanism:** <what the fix does>.
- **Detection signal:** <profile/log/config signature>; `<executable pointer>`.
- **Prevention/guard:** <assert/test/warning that catches this class, or the gap>.
- **Generalizes to:** <the pattern>; carries to <2–4 adjacent situations>.
```
