---
id: case-nccl-symmetric-graceful-fallback
type: instability-case
family: communication-negotiation
module: communication
maturity: full
instability_class: [opportunistic-collective-fallback]
signals: [eager-fallback-in-log, cross-rank-hang]
subsystems: [communication]
introduced_via: [new-feature]
phase: [any-phase]
patterns: [pattern-opportunistic-collective-fallback]
nvbugs: []
commits: ["4e10bf8950bf"]
success_prs: [11042]
failed_prs: []
---

# NCCL_SYMMETRIC needs graceful fallback on registration failure and graph capture

> Part of the [Communication instability cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** commit `4e10bf8950bf` · PR #11042 — nccl symmetric with
  graceful fallbacks. Foundation for three follow-ups, only one of which has a
  case here: [preallocation for autotuning](nccl-symmetric-preallocation-for-autotuning.md)
  (#11326). The other two hardened the same feature against *deterministic*
  failures and so are not instability cases — #11870 (long-context OOM hang,
  nvbug 5930934) and #12015 (segfault at library load on version mismatch,
  nvbugs 5923949 / 5803120); both were
  removed on 2026-08-12. Read their PRs directly if you are chasing a hang or a
  load-time crash on this path.
- **Symptom (variance signature):** NCCL_SYMMETRIC registered buffers to
  enable a fast zero-copy AllReduce; two failure modes made the resulting
  run unstable: (a) if buffer registration failed, the code path did not
  gracefully fall back to plain NCCL — the operation could error out or
  produce degraded / hung behaviour depending on the workload;
  (b) NCCL_SYMMETRIC buffer registration is *not permitted* during CUDA
  graph capture, so any tuner-triggered registration inside a captured
  region would fail.
- **Root cause:** the initial NCCL_SYMMETRIC integration assumed
  registration always succeeded and always ran outside graph capture;
  neither invariant is guaranteed, and there was no fallback wiring.
- **How introduced:** NCCL_SYMMETRIC landed as a new opportunistic fast
  path without a fallback ledger; graph-capture detection was missing.
- **Fix mechanism:** if a problem occurs during creation of registered
  tensors, perform the unregistered (plain) NCCL operation instead; also
  detect execution during graph capture (registration is not possible
  during capture) and skip registration for that call. Follow-up PR
  [#11326](nccl-symmetric-preallocation-for-autotuning.md) preallocates
  buffers before graph capture so the fallback fires less often.
- **Detection signal:** log messages describing NCCL_SYMMETRIC skipped or
  falling back to plain NCCL; `grep -nE 'NCCL_SYMMETRIC|graph.capture|register' bench.log`
  and `grep -n 'NCCL_SYMMETRIC' tensorrt_llm/_torch/distributed/`.
- **Prevention/guard:** every opportunistic fast collective must have an
  explicit, tested fallback path AND detect graph-capture state; a review
  checklist for adding opportunistic collectives should require both.
- **Generalizes to:** `pattern-opportunistic-collective-fallback`; carries
  to MNNVL / NVLS / symm-mem / userbuffers integrations, any collective
  requiring pre-registration, and any accelerator API that forbids
  side-effects inside a capture region.
