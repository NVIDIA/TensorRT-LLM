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

# A container-mounted $HOME poisoned the Triton cache; the "−22 % ctx_only regression" never reproduced

> Part of the [Measurement & test regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6368463` · commit `833ddd2a5903` · PR #15713 —
  adds `"--no-container-mount-home"` to `srunArgs` in
  `runLLMTestlistWithSbatch` (`jenkins/L0_Test.groovy`, +19/−0). **The PR body is
  empty and names no bug**; the linkage lives only in the NVBug, which records the
  `FileNotFoundError: [Errno 2] No such file or directory: '/root/.triton/cache'`
  failure as fixed by PR #15713.
- **Symptom:** reported as a `total_token_throughput` gap of −21.86 % — 8,727 →
  6,819 — on a NIXL disagg `ctx_only` case, B200 / <cluster>. **The reported gap did not
  reproduce, and the bug closed as not-reproduced.** Re-measured per the NVBug:
  good commit `[6969.3, 6891.82, 7160.39]` (median 6,969.3) vs bad
  `[6964.61, 6836.78, 6751.9]` — no gap at all, and note the *baseline itself*
  (8,727) is the unreproducible number, not the "bad" one. Record this case for the
  mechanism, not for a delta.
- **Root cause:** the CI job's `srun` mounted the submitting user's `$HOME` into the
  container, so container processes resolved `~/.triton/cache` (and every other
  dot-cache) to a **shared, host-side, cross-job** directory. That produces two
  distinct failures from one cause: hard errors when a cache entry is missing or
  half-written (`FileNotFoundError … '/root/.triton/cache'`), and *silent* timing
  changes when a run inherits or is denied another job's JIT artifacts. Either way
  the number measured is a property of the host's home directory state, not of the
  commit — which is precisely why the "regression" was one-shot and unreproducible.
- **How introduced:** `unknown`. No culprit commit; the mount behaviour is the
  cluster/job configuration, and the bug's reported baseline came from a job that
  happened to see a different cache state.
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
  or contended JIT cache is a documented multi-failure-mode hazard, biasing timings
  as often as it errors); and treat "reported baseline not reproducible" as a
  first-class triage outcome that closes the bug — chasing the delta forward would
  have burned a bisect on noise.
- **Generalizes to:** `pattern-measurement-not-product`; carries to every
  host-shared cache reachable from a container (`~/.triton`, `~/.cache/flashinfer`,
  DeepGEMM JIT, `~/.local` user-site packages, ccache), and to any perf gap whose
  good side is a *single historical* datapoint. Related: harness-side artifacts in
  this family, and the JIT-cache instability cases in the instability cookbook's
  warmup family.
