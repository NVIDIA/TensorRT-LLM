# Regression-Case Schema & Template

Every cookbook entry follows this schema. A regression case is a *diagnostic
precedent* — what regressed, how it was detected, why it happened, how it was
fixed, and how to catch the same class earlier next time. It is **not** an
optimization recipe (that is `perf-optimization-casebook`'s job) and not a
re-explanation of how a subsystem works.

Cases are distilled from NVBug-fix commits on TensorRT-LLM `main`. Every case
must be traceable: NVBug ID(s), commit hash(es), and PR number(s) are
mandatory provenance — a case without them cannot exist (there is nothing to
deep-dive later). This cookbook ships in the public TensorRT-LLM repository:
NVBugs are cited by ID only, and every other fact comes from a public PR
description, commit, diff, or GitHub issue. Never copy or paraphrase NVBug
descriptions, comments, titles, status, severity, or people.

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
type: regression-case
family: <tags.yaml::family>       # coarse how-it-broke LABEL only;
                                  # it does not name a directory
module: <dir name>                # data/modules.yaml; MUST equal this
                                  # file's parent directory name. Pick by
                                  # where the FIX landed, not where the
                                  # symptom showed.
maturity: stub | full             # full = distilled from PR description + diff;
                                  # stub = core slots roughly filled only
regression_class: [<tags.yaml::regression_class>]   # 1–2
signals: [<tags.yaml::signals>]   # 2–5 observable detection signals
subsystems: [<tags.yaml::subsystems>]
introduced_via: [<tags.yaml::introduced_via>]  # how the regression got in,
                                  # when the PR/diff makes it identifiable;
                                  # else [unknown]
phase: [prefill | decode | any-phase]
patterns: [<data/patterns.yaml ids>]  # EVERY pattern the case instantiates
nvbugs: ["<id>"]                  # ALL NVBug IDs this case covers, quoted
commits: ["<12-char hash>"]       # ALL fix commits on main, quoted
success_prs: [<PR numbers>]       # fix PRs that LANDED (merged on main). Every
                                  # hash in commits: comes from one of these.
failed_prs: [<PR numbers>]        # fix ATTEMPTS on the same nvbug(s) that did
                                  # NOT land: closed unmerged, rejected on
                                  # review, or still open but SUPERSEDED by a
                                  # merged fix in success_prs. [] when none.
                                  # An attempt still open and NOT superseded is
                                  # neither — leave it out until it merges or
                                  # closes. success_prs must be NON-EMPTY: a bug
                                  # with no merged fix has no case, and is not
                                  # recorded anywhere.
---
```

The human-readable bullets below remain the case body; frontmatter carries the
*matching* metadata plus the full provenance, bullets carry the *reasoning*.

## Field definitions (body bullets)

- **Provenance** — `nvbug <id> · commit <12-char hash> · PR #<n>` per fix,
  with the commit subject. When a bug was fixed by a series, list the primary
  first and the follow-ups as `related:` lines (fold-vs-new rule: a PR series
  fixing the *same* regression is ONE case).
- **Failed attempts** — one line per PR in `failed_prs:`: `PR #<n> — <what it
  tried> · <why it did not land>`. Omit the bullet entirely when `failed_prs`
  is `[]`. This is the highest-value half of the provenance for an agent about
  to propose a fix: it is the record of which plausible fixes were *tried and
  rejected*, so the next attempt does not re-propose one. Give the rejection
  reason from the PR's own review thread or closing comment — "reverted the
  wrong layer", "rejected on design grounds", "superseded by #<n>" — never a
  guess. A still-open attempt belongs here once a *different* PR merged the
  fix (say so: "still open, superseded by #<n>"); an open attempt that may yet
  land does not.
- **Symptom** — what regressed and where it was seen: the metric
  (throughput/TPS, TTFT, ITL/TPOT, memory→achievable batch, startup time),
  the model/config/hardware if the PR names them, and how it surfaced
  (perf CI bar, QA sweep, profile) when a public source says so.
- **Root cause** — the actual defect, in one or two lines, as established by
  the PR description/diff. Say what broke, not just where.
- **How introduced** — the change that caused the regression (commit/PR if the
  fix PR identifies it; else the code motion described, e.g. "refactor moved X
  out of the graphed region"; else "unknown — not stated in the PR").
- **Fix mechanism** — what the fix does, in one or two lines.
- **Detection signal** — how an agent would spot this class: the profile
  signature (nsys/ncu), log line, config check, or metric delta to look for.
  Include at least one **executable pointer** in backticks (a grep, a knob to
  inspect, a profile view) — not only a described observation.
- **Prevention/guard** — what would catch this class earlier: the assert,
  unit/perf test, log warning, or review checklist item. If the fix PR added
  such a guard, name it; else state the gap.
- **Generalizes to** — the transferable regression *pattern* this instantiates
  and 2–4 adjacent situations where the same failure mode can recur. Match on
  this, not the title.

## Anti-fabrication rules

- Every number, knob name, and file path must be traceable to a public
  source: the fix PR's description or diff, a commit message, or a GitHub
  issue. If you didn't look it up, don't write it.
- Quantitative regression sizes ("-15% TPS") appear ONLY when a public PR
  description/title, commit message, or GitHub issue states them; cite which
  one. Never take a number, bisect result, root cause or verification from an
  NVBug.
- Model/hardware applicability comes from the PR text or diff, never inferred
  from resemblance.
- If the PR description is too thin to fill Root cause + Fix mechanism at
  least roughly, the entry belongs in `data/pending.yaml`, not here.

## Template

Each case is its **own file** at `references/<module>/<slug>.md`: an H1 title
(short, states the regression, not the fix), a one-line breadcrumb back to the
module index and this schema, then every field as a bold bullet:

```
---
<frontmatter — copy the skeleton above; canonical terms from data/tags.yaml>
---

# <short title naming the regression>

> Part of the [<Module display name> regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `<id>` · commit `<12-char hash>` · PR #<n> — <subject>;
  related: nvbug `<id>` · commit `<hash>` · PR #<n> — <subject>.
- **Failed attempts:** PR #<n> — <what it tried> · <why it did not land>.
  (omit this bullet when `failed_prs: []`)
- **Symptom:** <metric that regressed> on <model/config/HW if stated>;
  surfaced via <perf CI / QA / profile, if a public source says so>.
- **Root cause:** <the defect>.
- **How introduced:** <causing change, or "unknown — not stated in the PR">.
- **Fix mechanism:** <what the fix does>.
- **Detection signal:** <profile/log/config signature>; `<executable pointer>`.
- **Prevention/guard:** <assert/test/warning that catches this class, or the gap>.
- **Generalizes to:** <the pattern>; carries to <2–4 adjacent situations>.
```
