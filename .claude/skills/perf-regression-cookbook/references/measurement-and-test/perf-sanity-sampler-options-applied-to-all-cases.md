---
id: case-perf-sanity-sampler-options-applied-to-all-cases
type: regression-case
family: measurement-and-test
module: measurement-and-test
maturity: full
regression_class: [measurement-artifact]
signals: [perf-ci-bar-failure, throughput-drop]
subsystems: [perf-test-config, sampler]
introduced_via: [new-feature]
phase: [any-phase]
patterns: [pattern-measurement-not-product]
nvbugs: ["5666804"]
commits: ["d2327095689a"]
success_prs: [9512]
failed_prs: []
---

# perf-sanity applied top_k/top_p/temperature to every case, so unrelated tests measured a sampler they never asked for

> Part of the [Measurement & test regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5666804` · commit `d2327095689a` · PR #9512 —
  "[https://nvbugs/5666804][test] only adding sampler config for limited
  models".
- **Symptom:** a broad, simultaneous perf-sanity drop across cases that share
  nothing but the harness. The diff is the evidence, and there is **no per-case
  percentage** in the PR — do not quote one.
- **Root cause:** `get_sampler_options_config()` in the perf-sanity harness
  returned a non-empty dict — `{'top_k': 4, 'top_p': 0.5, 'temperature': 0.5}` —
  for **every** test label, and the command builder appended
  `--sampler_options <json>` unconditionally. So a case whose intent was greedy
  decoding silently ran top-k/top-p sampling. That changes the sampler code path
  (and therefore host work per step) for the entire suite, which is why the
  regression looks like an engine change while every engine number is innocent.
  The defect shape is the one that matters here: a lookup helper that *defaults to
  a value* instead of defaulting to "nothing to add".
- **How introduced:** `new-feature` — sampler-options support was added for a
  handful of cases that genuinely need it, and the "which cases" half was
  effectively `True`.
- **Fix mechanism:** narrow the helper to an explicit inline allowlist of 9 test
  labels and make the append conditional: `if sampler_config:` before adding
  `--sampler_options`. Nothing about the sampler changes — only *who gets it*. Note
  the allowlist is inline in the helper rather than in a YAML: it is a harness
  behaviour, and keeping it next to the code that consumes it is what makes the
  next reader see the coupling.
- **Detection signal:** in a run log,
  `grep -n "sampler options config:" <log>` and
  `grep -n -- "--sampler_options" <log>` — a case that is supposed to be greedy
  showing either line is this defect. Statically:
  `git grep -n "sampler_options" tests/integration/defs/perf/` and check that every
  producer of that flag is gated on a case identity.
- **Prevention/guard:** no test was added. The rule this case exists to teach:
  **a harness helper that returns test-affecting configuration must return empty by
  default**, and the caller must treat empty as "append nothing". An unconditional
  append is undetectable from any single case's result — it only shows up as a
  suite-wide shift, which is exactly the signature that gets misfiled as a product
  regression.
- **Generalizes to:** `pattern-measurement-not-product`; carries to every
  harness-side knob applied by default (`--warmup`, `--concurrency`,
  `--ignore-eos`, env injection, container args) and to the broader rule that when
  many unrelated perf cases move together, **suspect the harness first** and diff
  `jenkins/scripts/perf/**` + `tests/integration/defs/perf/**` at the commit under
  test before bisecting the engine.
