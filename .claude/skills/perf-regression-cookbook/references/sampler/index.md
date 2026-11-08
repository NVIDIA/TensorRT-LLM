# Regression Cookbook — Sampler

This module is the sampling layer: `TorchSampler` and its option handling, the
stop-words / logprobs machinery, beam search, and the host↔device handoff that
turns device-side logits into request state every decode step. It runs once per
iteration for the whole batch, so a small per-step cost here is charged to every
token, and — because the sampler is where device results become Python objects —
it is the most common place for an *implicit* device→host copy to appear. First
thing to check on a decode-side regression with kernel durations unchanged: what
the sampler does per step, and whether anything new in it reads a device tensor
on the host.

## Recurring patterns in this module

- **Per-step sync added** — an implicit `.item()`/`.tolist()`/comparison against a
  device tensor forces a synchronization every iteration; the trace shows GPU idle
  waiting on the host with kernel durations unchanged. Both cases here are that
  shape — stop-words checked against device state, and blocking D2H copies issued
  on the worker thread. Keep the check device-side, or batch it.
  _(Instances: the stop-words implicit D2H sync; the CC blocking D2H copies on the
  worker thread.)_
- **Host work on the hot path** — sampler-side Python work paid every step even
  when the feature is inert: assembling logprobs structures when zero logprobs
  were requested, and a beam-search handoff performed on the host. Guard on the
  request actually asking for the feature, and measure host span, not kernels.
  _(Instances: the TorchSampler logprobs=0 overhead; the beam-search host
  handoff.)_
- **Optimization lost in a port** — sampler optimizations that existed on one
  branch did not survive a port/merge, so the perf property silently reverted with
  no functional change and no new slow code to find. When a known optimization
  stops showing up, diff the branch that had it rather than bisecting for a
  culprit.
  _(Instance: the TorchSampler optimizations lost in a branch.)_

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [Beam search performs its handoff on the host](beam-search-host-handoff.md) | TinyLlama L40S beam_width=10: TTFT −2.13% mean after #13748; TTFT −0.092 ms / E2E −0.448 ms median after #13799 | host-work-added |
| [CC blocking D2H copies issued on the worker thread](cc-blocking-d2h-copies-worker-thread.md) | CC-only stall; the case states no percentage, model or SKU | sync-introduced |
| [Stop-words handling introduced an implicit D2H sync per step](sampler-stop-words-implicit-d2h-sync.md) | ~3× throughput drop, 10,607.09 → 4,433.11 tok/s; culprit `02edb19f4302` | sync-introduced |
| [TorchSampler paid logprobs assembly cost with logprobs=0](torchsampler-logprobs0-overhead.md) | Llama-3.2-1B L40S bs 1000: 2.4 ms GPU / 13.8 ms host per step → 1.0 / 11.3 after the fix | host-work-added |
| [TorchSampler optimizations did not survive a branch port](torchsampler-optimizations-lost-in-branch.md) | the optimized sampler path is absent on the branch; no quantitative delta stated | optimization-lost-in-port |
