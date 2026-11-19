---
id: case-moe-deepep-gc-hang
type: instability-case
family: runtime-determinism
module: moe
maturity: full
instability_class: [gc-driven-collective-destruction]
signals: [cross-rank-hang, rep-to-rep-variance]
subsystems: [moe, communication]
introduced_via: [new-feature]
phase: [any-phase]
patterns: [pattern-gc-driven-collective-destruction]
nvbugs: []
commits: ["e22693019404"]
success_prs: [12060]
failed_prs: []
---

# DeepEP `Buffer.__del__` calls a collective — non-deterministic GC deadlocks ranks

> Part of the [MoE instability cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** JIRA `TRTLLM-11037` · commit `e22693019404` ·
  PR #12060 — Fix MoE DeepEP hang caused by non-deterministic GC
  (MERGED to `main` 2026-03-13, mergeCommit
  `e2269301940457277b97cd0589ca2443d52f4ce2`). No NVBug: the PR is
  JIRA-tracked and its body names none, so `nvbugs: []` is correct
  rather than incomplete.
- **Symptom (variance signature):** multi-GPU MoE tests using DeepEP /
  DEEPEPLOWLATENCY on H100 hung at rank drain — sometimes at the end of
  a test, sometimes during fallback from DeepEP to
  AllGatherReduceScatter mid-run. Reproduction depended on GC timing,
  so the hang appeared "flaky" on CI and got the tests skipped rather
  than diagnosed. Same input, same seed, same nodes could either finish
  or hang.
- **Root cause:** DeepEP's `Buffer.__del__` calls
  `intranode::barrier`, a **collective** op that requires every rank
  to enter the barrier before any rank can leave. Python garbage
  collection of `Buffer` instances is not synchronised across ranks:
  when rank A's GC happens to run before rank B's, rank A enters the
  barrier and blocks forever waiting for rank B — which will only get
  there when its own GC decides to run. Some rank pairs eventually
  converge; others do not.
- **How introduced:** DeepEP was integrated as an opportunistic MoE
  fast path without a per-rank synchronous release primitive; the
  finalizer's implicit collective was accepted from the upstream
  library without an explicit destroy hook.
- **Fix mechanism:** add an explicit `destroy()` method on the
  `Communication` base class, implemented by `DeepEP` and
  `DeepEPLowLatency` to release the buffer synchronously (all ranks
  hit the barrier at the same well-defined point). Call `destroy()`
  from `ConfigurableMoE` on the DeepEP → AllGatherReduceScatter
  fallback path AND make `ConfigurableMoE` a context manager
  (`with create_moe(...) as fused_moe:`) so scope exit releases
  resources deterministically. Unit tests updated to the context-
  manager pattern. Re-enable H100 DEEPEPLOWLATENCY multi-GPU tests
  that had been skipped for this hang.
- **Detection signal:** multi-GPU test hangs at teardown with no
  Python traceback; `py-spy dump` on the stuck process shows the
  main thread inside a `__del__` calling a barrier / collective;
  `grep -nE 'intranode::barrier|Buffer.__del__|destroy\(\)' tensorrt_llm/_torch/modules/fused_moe/`
  should show the explicit destroy path is used everywhere the buffer
  is released.
- **Prevention/guard:** finalizers / destructors of objects held
  across ranks must NOT contain implicit collective operations. If a
  collective is required to release, expose an explicit `destroy()`
  and require callers to invoke it at a well-defined program point
  (context manager scope, explicit close call). A CI check that
  greps for `__del__` implementations calling barrier / allreduce /
  collective primitives.
- **Generalizes to:** `pattern-gc-driven-collective-destruction`;
  carries to any Python binding wrapping a distributed C++ resource
  whose destructor synchronises across ranks (nccl comms, MPI
  windows, symmetric-memory pools, cutlass workspaces, TransformerEngine
  RS buffers), and to any lifecycle pattern where per-rank timing
  variance drives a collective's liveness.
