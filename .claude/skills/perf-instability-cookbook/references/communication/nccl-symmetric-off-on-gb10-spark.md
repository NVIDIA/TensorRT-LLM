---
id: case-nccl-symmetric-off-on-gb10-spark
type: instability-case
family: communication-negotiation
module: communication
maturity: full
instability_class: [platform-conditional-disable]
signals: [cross-rank-hang, platform-specific-failure]
subsystems: [communication]
introduced_via: [incomplete-coverage]
phase: [any-phase]
patterns: [pattern-platform-conditional-disable]
nvbugs: []
commits: ["cd650702c70c"]
success_prs: [12902]
failed_prs: []
---

# NCCL_SYMMETRIC AllReduce fails on GB10 (DGX Spark) — targeted per-platform disable

> Part of the [Communication instability cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** GitHub issue `#12715` · commit `cd650702c70c` ·
  PR #12902 — disable NCCL_SYMMETRIC tactic on GB10 (DGX Spark). Also
  companion to
  [case-nccl-symmetric-graceful-fallback](nccl-symmetric-graceful-fallback.md)
  (#11042).
- **Symptom (variance signature):** on **dual GB10 (DGX Spark)** hosts
  (SM121, 128 GB unified memory, inter-node TP/EP over 200 GbE RoCE),
  NCCL_SYMMETRIC AllReduce hits a NCCL-internal failure. Issue #12715
  documents **two** manifestations and no others: a **segfault** inside
  `ncclGinGdakiQueryLastError` → `ncclCommGetAsyncError` →
  `AllreduceOp::run` during autotuner warmup, and — once
  `NCCL_GIN_ENABLE=0` suppresses that — an **indefinite hang**
  immediately after MNNVL probing resolves to `finalMNNVL=0`, never
  proceeding to the plain NCCL IB fallback. There is **no report of an
  incorrect reduction**; do not cite one. The issue's A/B is by
  release, not by host: Qwen3-235B-A22B-FP4 and
  Nemotron-3-Super-120B-A12B-NVFP4 at `--tp_size 2` work on 1.2.0rc6
  and fail on 1.3.0rc10 with identical configuration. Runs that touch
  the NCCL_SYMMETRIC tactic on this platform never complete; without
  the tactic they run fine. PR #12902's own scope claim is only that
  the WAR lands "without impacting other platforms" — the specific SKUs
  said to be unaffected are not enumerated in either the issue or the
  PR, so treat "everything but GB10" as the evidenced bound rather than
  a validated per-SKU matrix.
- **Root cause:** an unresolved bug in NCCL on GB10 (DGX Spark). Not a
  TRT-LLM bug — TRT-LLM is only the consumer of the failing NCCL
  tactic.
- **How introduced:** the NCCL_SYMMETRIC tactic was integrated for the
  cluster classes NVIDIA validated against; the GB10 (Spark)
  configuration was not covered by that validation matrix.
- **Fix mechanism:** conditionally disable the NCCL_SYMMETRIC tactic on
  GB10 SKUs at AllReduce tactic-selection time. Explicit
  workaround / WAR — to be removed once the upstream NCCL fix lands.
  Companion cleanup improves the tactic-cache-miss fallback path so
  the disabled tactic degrades cleanly to plain NCCL.
- **Detection signal:** on GB10 with `NCCL_SYMMETRIC` enabled, either a
  segfault whose backtrace names `AllreduceOp::run` under
  `ncclCommGetAsyncError` during autotuner warmup, or a hang whose last
  log line is the MNNVL probe
  (`localNVLink=1, localFabricValid=0, allRanksSameFabric=0, finalMNNVL=0`);
  the same job on a non-Spark host completes. Note that the NCCL env
  knobs a reader would reach for first —
  `NCCL_MNNVL_ENABLE=0`, `NCCL_GIN_ENABLE=0`, `NCCL_IB_MERGE_NICS=0`,
  `NCCL_SYMMETRIC_ENABLE=0`, `TRTLLM_ALLREDUCE_STRATEGY=NCCL` — were all
  tried on #12715 and **none resolved the hang**, which is why the fix
  had to be a code-side tactic gate.
  `nvidia-smi -q | grep -E 'Product Name|SKU'` reports GB10 (Spark);
  `grep -nE 'GB10|Spark|is_gb10' tensorrt_llm/_torch/distributed/`
  should show the per-platform gate.
- **Prevention/guard:** any opportunistic collective must have a
  per-platform allowlist / blocklist knob AND a documented mechanism
  for adding a platform. A CI matrix that covers every supported SKU
  before a fast path defaults on.
- **Generalizes to:** `pattern-platform-conditional-disable`; carries
  to any collective whose failure is SKU-specific (MNNVL on non-
  NVLink-domain hosts, userbuffers on hosts without the required
  driver, NVLS on hosts without the switch), and to any accelerator-
  library WAR that must live in TRT-LLM until an upstream library fix
  ships. The pattern is a signpost for "when the upstream fix lands,
  delete this gate."
