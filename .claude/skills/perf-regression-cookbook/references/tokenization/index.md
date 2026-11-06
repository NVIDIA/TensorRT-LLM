# Regression Cookbook — Tokenization & detokenization

This module is the tokenizer layer around the engine: tokenizer construction and
the per-model tokenizer shims/wrappers, plus the encode/decode work that runs
inside request intake and response assembly. It is pure host code sitting in front
of the forward pass, so a slow path here costs TPOT and throughput while leaving the
GPU kernel inventory *completely unchanged* — which is exactly what makes it easy to
misdiagnose. First thing to check when throughput collapses but the profile looks
identical: the host spans in request intake (`_fetch_new_requests`,
`broadcast_requests`) and which tokenizer class was actually constructed for this
model.

## Recurring patterns in this module

- **Host work on the hot path** — a per-model tokenizer shim takes a slow
  (non-fast-tokenizer) code path and pays it on every request, inside the intake
  step of the serving loop. The GPU is idle waiting on the host; kernels and their
  durations are unchanged, so compare host spans rather than a kernel summary.
  Worth knowing that this one is **superseded at HEAD** — PR #14741 deleted the shim
  entirely, so the perf property now holds structurally rather than by care.
  _(Instance: the Kimi tokenizer slow path on the hot path.)_

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [Kimi tokenizer shim takes the slow path on every request](kimi-tokenizer-slow-path-on-hot-path.md) | text-only `k25_thinking_fp4` (ISL=8K, conc 2): TPOT 9.8× higher, throughput −92%; `_fetch_new_requests`/`broadcast_requests` +~34%, GPU kernel inventory unchanged | host-work-added |
