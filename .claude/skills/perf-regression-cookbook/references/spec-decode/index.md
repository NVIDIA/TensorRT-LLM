# Regression Cookbook — Speculative decoding

This module is the speculative-decoding machinery: MTP / draft-model decoding, the
draft-token bookkeeping that maps tokens back to requests, and the small
device-side helpers (argmax and friends) that run per draft step. It has a failure
mode no other module has: a defect can leave every kernel exactly as fast as before
and still cost throughput, by lowering the **acceptance length** — fewer accepted
draft tokens means more iterations for the same output. First thing to check on a
spec-decode regression: the reported acceptance length at the affected batch size,
before looking at any kernel. If acceptance moved, stop profiling and audit the
draft bookkeeping; if it did not, the defect is an ordinary kernel/host cost.

## Recurring patterns in this module

- **Kernel swap regressed** — a per-draft-step helper was replaced with a new
  implementation that is slower at the shapes MTP actually produces, and the loss
  is multiplied by the draft length. Measure the swapped helper at the real per-step
  shape; the revert is a legitimate fix here.
  _(Instance: the CuteDSL argmax revert.)_
- **Stale metadata across a layout change** — a cached token-to-request map is not
  invalidated when the layout it indexes changes, so draft tokens are attributed to
  the wrong request and acceptance collapses. The signature is batch-size
  dependence: correct at batch 1, worse as batch grows — because the map only
  becomes wrong when there is more than one request to confuse. Kernels and GPU
  time look normal throughout.
  _(Instance: the DSA MTP stale token-to-request map.)_

_Note, carried from the old index: an EAGLE3 variant-misroute case (`topK=1`
routed to the dynamic-tree path, nvbug 6394425) was removed on 2026-08-12. No
confirmed case in this module covers EAGLE3 path selection._

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [CuteDSL argmax slower on the MTP draft path — revert](cutedsl-argmax-revert.md) | spec-decode perf regression after the CuTe-DSL argmax swap (PR #10476); the fix PR states no metric | kernel-swap-regressed |
| [DSA MTP stale token-to-request map lowers acceptance length](dsa-mtp-stale-token-to-request-map.md) | GLM-5.2 NVFP4 GB200 MTP k=7: acceptance length 2.789 → 3.909 (+40.2%) with the fix at conc 64 / batch 16; normal at batch 1, worse as batch grows | stale-cached-metadata |
