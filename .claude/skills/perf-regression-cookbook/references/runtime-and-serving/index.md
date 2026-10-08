# Regression Cookbook — Runtime & serving

This module is the general runtime plumbing under the model: the buffer/memory
pool and its allocation helpers, the custom-op registration and dispatch layer,
device-capability detection, and the serving-side request/response handling around
a generation (including media encode/decode on the response path). Nothing here
is model-specific, which is why a defect in it shows up on whatever workload
happens to exercise it hardest — a per-call dispatch tax matters at thousands of
ops per step, a buffer zero-fill matters when the buffers are large, an encode on
the response path matters per request. First thing to check: whether the added
cost is per-call host time, per-request host time, or extra device work, since the
three have different signatures and only the last is visible in a kernel summary.

## Recurring patterns in this module

- **Host work on the hot path** — per-call or per-request host cost that the
  device profile does not show, because kernel names and durations are unchanged.
  Two shapes here: a custom-op dispatcher tax paid on every op invocation
  (~7 µs × ~2100 calls/step), and an image encode executed on the serving
  response path. Compare host spans, and note that the dispatch shape only hurts
  where CUDA graphs are *not* capturing the ops away — which is why it appeared at
  8 GPUs without graphs and nowhere else.
  _(Instances: the LTX-2 custom-op dispatch tax; PNG encode on the serving hot
  path.)_
- **Redundant device work** — a generic fill/initialization helper zeroes buffers
  that are about to be fully written, so real kernels appear in the profile doing
  work no one needs. The tell is a fill/memset kernel with a large share of a
  phase; check whether the consumer overwrites the whole buffer.
  _(Instance: the buffer-pool zero-fill FillFunctor.)_
- **Fast path silently fell back** — capability detection is a routing input:
  treating a device attribute as equivalent to compute capability sends a device
  down a path meant for another, correctly and slower. Verify what the detection
  reports on the SKU in question rather than what it is expected to report.
  _(Instance: NVLE-only treated as CC.)_

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [Buffer-pool FillFunctor zero-fills buffers that are fully overwritten](buffer-pool-zero-fill-fillfunctor.md) | 4 large FillFunctor kernels account for 43% of one ctx block; the PR body states no throughput number | redundant-device-work |
| [Custom-op dispatch tax on the LTX-2 hot path](custom-op-dispatch-tax-ltx2.md) | LTX-2 NVFP4 E2E +19.5% (9.95 s → 11.89 s) on 8×B200, only without CUDA graphs; ~7 µs/call × ~2100 calls/step ≈ 11.3 ms/step | host-work-added |
| [NVLE-only device treated as compute-capability, routing to the wrong path](nvle-only-treated-as-cc.md) | structural misroute; the case states no metric, model, hardware or percentage | fast-path-fallback |
| [PNG encode executed on the serving hot path](png-encode-on-serving-hot-path.md) | ~1.5 s extra per image request; b64_json 12.71 s → 10.78 s (−1.93 s / 15.2%) on FLUX.2-dev/B200 | host-work-added |
