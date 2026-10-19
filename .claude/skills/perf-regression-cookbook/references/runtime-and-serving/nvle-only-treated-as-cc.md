---
id: case-nvle-only-treated-as-cc
type: regression-case
family: execution-and-graph
module: runtime-and-serving
maturity: full
regression_class: [fast-path-fallback]
signals: [throughput-drop, host-time-increase]
subsystems: [runtime-python, autotuner, scheduler-executor]
introduced_via: [prior-fix-side-effect]
phase: [any-phase]
patterns: [pattern-fast-path-silent-fallback]
nvbugs: ["6506990"]
commits: ["6ccdb019cd33"]
success_prs: [16850]
failed_prs: []
---

# NVLE-only systems were misdetected as confidential compute, so three fast paths disabled themselves

> Part of the [Runtime & serving regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6506990` · commit `6ccdb019cd33` · PR #16850 —
  distinguishes NVLE from confidential compute in the platform-detection helper.
- **Symptom:** **the PR states no metric, no model, no hardware and no
  percentage.** Its entire symptom statement is structural: "NVLE can be enabled
  independently of CC, so CC restrictions such as H2D bounce buffering do not apply
  to NVLE-only systems." Record it that way — the value of this case is the blast
  radius and the detection probe, not a delta. Do not attach a number to it.
- **Root cause:** one boolean stood for two independent platform features.
  `confidential_compute_enabled()` returned true when **either** CC **or** NVLE
  (NVLink Encryption) was active, and three consumers then applied CC's
  restrictions to NVLE-only machines: the pinned-memory decision in `_utils.py`,
  `_use_global_timer` in `autotuner.py`, and `enable_async_worker` in
  `pyexecutor/_util.py`. So an NVLE-only box paid CC's costs — bounce-buffered
  copies, the async-worker sampler path, a different autotuner timer — for a
  restriction that does not apply to it. This is the silent-fallback shape in its
  purest form: every consumer is *correct given its input*, the input is simply an
  over-broad predicate, and nothing logs the downgrade.
- **How introduced:** `prior-fix-side-effect`. The helper was introduced by
  PR #8463 (see `case-cc-blocking-d2h-copies-worker-thread`), which needed exactly
  one bit — "must D2H copies be blocking?" — and NVLE was folded in because it
  shares part of the NVML surface. The lesson is at the API boundary: a predicate
  named for one feature that returns true for two is a latent misroute for every
  future consumer, and consumers arrive later than the predicate.
- **Fix mechanism:** replace the single boolean with
  `@lru_cache(maxsize=1) get_cc_and_nvle_status() -> tuple[bool, bool]`, returning
  the two states separately, and update each of the three call sites to consume the
  one it actually means. The bare `except` clauses around the NVML probes are
  narrowed to `pynvml.NVMLError`. Note the `lru_cache`: the status is read once per
  process, which is right for a hardware property but means an env-var override
  applied after first call is ignored.
- **Detection signal:** on the machine in question,
  `python -c "from tensorrt_llm._utils import get_cc_and_nvle_status as f; print(f())"`
  → `(False, True)` is an NVLE-only system, and on a pre-fix tree
  `python -c "from tensorrt_llm._utils import confidential_compute_enabled as f; print(f())"`
  returns `True` there — that disagreement *is* the bug. Statically:
  `git grep -n "confidential_compute_enabled" tensorrt_llm/` and check, per hit,
  whether the restriction being applied is a CC restriction or an encryption one.
- **Prevention/guard:** a real guard was added — `tests/unittest/utils/
  test_confidential_compute.py` (+172), registered in `l0_cpu.yml` so it runs
  without a GPU, covering the CC / NVLE / both / neither matrix. That is the right
  shape for a platform predicate: the combinations are enumerable, so enumerate
  them. Rule: never let one boolean answer two platform questions, and if a
  predicate's name implies feature X, it must be false when only feature Y is
  present.
- **Generalizes to:** `pattern-fast-path-silent-fallback`; carries to every
  capability probe that gates a fast path (MNNVL, NCCL symmetric memory, P2P
  availability, CC/NVLE, IOMMU) and to the general failure mode where a
  **conservative default applied to the wrong platform** costs performance with no
  log line, no error and no test. When a machine is slower than its siblings for no
  visible reason, enumerate the capability predicates before profiling kernels.
