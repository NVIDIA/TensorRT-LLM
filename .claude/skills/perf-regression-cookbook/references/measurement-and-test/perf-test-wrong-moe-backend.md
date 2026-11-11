---
id: case-perf-test-wrong-moe-backend
type: regression-case
family: measurement-and-test
module: measurement-and-test
maturity: full
regression_class: [measurement-artifact]
signals: [perf-ci-bar-failure, throughput-drop, slower-kernel-in-trace]
subsystems: [perf-test-config, moe]
introduced_via: [pre-existing-gap]
phase: [any-phase]
patterns: [pattern-measurement-not-product]
nvbugs: ["5830877"]
commits: ["a0e8ef74733f"]
success_prs: [11046, 11636]
failed_prs: []
---

# GPT-OSS 120B perf test ran the non-recommended CUTLASS MoE backend

> Part of the [Measurement & test regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5830877` · commit `a0e8ef74733f` · **two PRs, two
  branches — cite #11046 first.**
  - PR **#11046** "[https://nvbugs/5830877][fix] Use the best (correct)
    config for GPTOSS perf test" — merged **2026-02-05** onto
    **`release/1.2`** (merge commit `46b890af7799`, *not* on `main`). This
    is the only PR a `gh pr list --search "5830877"` lookup returns.
  - PR **#11636** "[None][chroe] Mass integration of release/1.2 - 5th" —
    merged **2026-02-24** onto **`main`**; this is the carryover that put
    the fix on `main` as commit `a0e8ef74733f` ("…GPTOSS perf test
    (#11046)", single file `tests/integration/defs/perf/pytorch_model_config.py`).
    Its title carries no bug id, so a `main`-side PR lookup and a
    bug-id PR search name different numbers for the same defect.
- **Symptom:** GPT-OSS 120B FP4 max-throughput perf tests reported
  sub-best throughput. The fix touches the max-throughput entry
  (`max_batch_size: 720` in the diff). The PR states the perf testing "wasn't using the recommended
  settings" — the product had not regressed, the measurement had.
- **Root cause:** The GPT-OSS 120B max-throughput entry in
  `tests/integration/defs/perf/pytorch_model_config.py` set
  `moe_config: {backend: 'CUTLASS'}`, but per the PR the TRTLLM MoE
  backend is the recommended setting for this model on Blackwell and
  "only this MOE backend's perf is guaranteed to be the best". The test
  therefore benchmarked (and gated on) a slower non-recommended path.
- **How introduced:** the fix PR does not name a regressing commit; git
  history (`git log -S "'backend': 'CUTLASS'"`) shows the CUTLASS setting
  was there from the test's introduction in PR #7328 (commit
  `e6073b3911`) — the perf test was never aligned with the recommended
  config.
- **Fix mechanism:** one-line config flip in
  `tests/integration/defs/perf/pytorch_model_config.py`:
  `moe_config.backend` `'CUTLASS'` → `'TRTLLM'` for the GPT-OSS 120B
  max-throughput test patterns, so the perf bar measures the recommended
  (guaranteed-best) backend.
- **Detection signal:** before bisecting product code for a tripped
  GPT-OSS/MoE perf bar, diff the test's config overrides against the
  model's recommended deployment settings:
  `grep -n -A3 "moe_config" tests/integration/defs/perf/pytorch_model_config.py`;
  in an nsys trace the wrong backend shows up as different MoE kernels
  than a correctly configured serve run.
- **Prevention/guard:** none added by the fix — the recommended backend
  is hand-pinned per test entry, so it can silently drift again when the
  recommendation changes. Gap: a periodic cross-check of perf-test
  `moe_config`/backend overrides against the model's recommended
  deployment config (or deriving the perf-test config from it) would
  catch the next mismatch at review time.
- **Generalizes to:** `pattern-measurement-not-product` — the perf bar
  tripped because the *test config*, not the product, was wrong. Carries
  to: any perf test that hand-pins a backend/kernel knob which later
  stops being the recommended one; benchmark harness defaults diverging
  from the deployment-guide config for the same model/hardware; perf
  bars established on a non-optimal path that then mask (or fake) later
  regressions; and the adjacent shape where the measured *regime* rather than
  the backend is wrong — an AutoDeploy llama bar measuring host-bound low
  concurrency (nvbug 6192201), which had a case here until 2026-08-12.
