---
id: case-nixl-ctx-only-gap-not-reproduced-home-mount
type: regression-case
family: measurement-and-test
module: measurement-and-test
maturity: full
regression_class: [measurement-artifact]
signals: [perf-ci-bar-failure, throughput-drop]
subsystems: [perf-test-config, build-dependency]
introduced_via: [unknown]
phase: [prefill]
patterns: [pattern-measurement-not-product]
nvbugs: ["6368463"]
commits: ["833ddd2a5903"]
success_prs: [15713]
failed_prs: []
---

# A container-mounted $HOME poisoned the Triton cache; the ctx_only "regression" never reproduced

> Part of the [Measurement & test regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6368463` · commit `833ddd2a5903` · PR #15713 —
  adds `"--no-container-mount-home"` to `srunArgs` in
  `runLLMTestlistWithSbatch` (`jenkins/L0_Test.groovy`, +19/−0), plus a
  `// TODO: Add mounts for different cache directories like pip, triton, etc.`
  in `getMountListForSlurmTest`. **The PR body is empty and its title carries
  `[None]`**, so nothing public links it to the bug id.
- **Symptom:** a `total_token_throughput` gap on a NIXL disagg `ctx_only` case
  that does not survive a clean re-measure. Record this case for the mechanism,
  not for a delta.
- **Root cause:** the CI job's `srun` mounted the submitting user's `$HOME` into the
  container, so container processes resolved `~/.triton/cache` (and every other
  dot-cache) to a **shared, host-side, cross-job** directory. That produces two
  distinct failures from one cause: hard errors when a cache entry is missing or
  half-written (e.g. a `FileNotFoundError` on `/root/.triton/cache`), and *silent* timing
  changes when a run inherits or is denied another job's JIT artifacts. Either way
  the number measured is a property of the host's home directory state, not of the
  commit — which is precisely why such a "regression" can be one-shot and
  unreproducible.
- **How introduced:** `unknown`. No culprit commit; the mount behaviour is the
  cluster/job configuration, not product code.
- **Fix mechanism:** stop mounting home — pass `--no-container-mount-home` on the
  sbatch/srun path, so each job's caches live inside the container. Nothing about
  NIXL, disagg, or the engine changes.
- **Detection signal:** `grep -n "no-container-mount-home" jenkins/L0_Test.groovy`
  — absent ⇒ pre-fix. In a failing log,
  `grep -nE "\.triton/cache|\.cache/(flashinfer|deep_gemm)|Errno (2|116)" <log>`.
  The methodological signal is the important one: **a single-shot gap whose
  *baseline* cannot be re-measured is a measurement artifact until proven
  otherwise**, so re-run the good commit before bisecting anything.
- **Prevention/guard:** no test — this is a job-configuration fix. Two rules:
  never let a container inherit a shared writable `$HOME` on a perf job (a poisoned
  or contended JIT cache is a multi-failure-mode hazard, biasing timings
  as often as it errors); and treat "reported baseline not reproducible" as a
  first-class triage outcome — chasing the delta forward burns a bisect on noise.
- **Generalizes to:** `pattern-measurement-not-product`; carries to every
  host-shared cache reachable from a container (`~/.triton`, `~/.cache/flashinfer`,
  DeepGEMM JIT, `~/.local` user-site packages, ccache), and to any perf gap whose
  good side is a *single historical* datapoint. Related: harness-side artifacts in
  this family, and the JIT-cache instability cases in the instability cookbook's
  warmup family.
