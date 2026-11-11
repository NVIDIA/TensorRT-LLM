# Regression Cookbook — KV-cache transceiver

This module is the disaggregated KV-transfer layer: the cache transceiver on the
context and generation servers, its handshake/quorum logic, and the request
hand-off work that runs on the generation side before the first token. It sits
directly in front of TTFT, so anything added here is paid before any token is
emitted — and because the layer is *distributed*, a correctness fix can add a
collective at a wider scope than the thing it guards and be charged to every
iteration. First thing to check on a disagg TTFT regression: what the generation
server does per request before the forward pass (re-tokenization, copies), and
whether any per-iteration collective was introduced in the window.

## Recurring patterns in this module

- **Host work on the hot path** — request hand-off work on the generation server
  is pure host cost in front of the first token: re-tokenizing a prompt the
  context server already tokenized, plus quadratic copying in the reuse path.
  _(Instance: the disagg gen server retokenize + O(N²) memcpy.)_
- **Per-step sync added** — a deadlock/correctness fix added a quorum vote over
  `MPI_COMM_WORLD` rather than over the ranks it actually guards, so every
  iteration pays a world-scope collective. When a fix in this layer is followed
  by a broad TTFT/TPOT shift, look for a new collective and check its
  communicator scope before looking at kernels.
  _(Instance: the disagg cache-transfer quorum vote at WORLD scope.)_

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [Disagg gen server retokenizes prompts and pays O(N²) memcpy in KV reuse](disagg-gen-retokenize-and-memcpy.md) | high TTFT on the Harmony GPT-OSS chat path; two host costs before the first token | host-work-added |
| [Disagg cache-transfer quorum vote over `MPI_COMM_WORLD` blocks every iteration](disagg-quorum-vote-world-scope.md) | GB200 disagg TTFT +57%, TPOT +7.6%; `benchmark_duration_s` 127.1 → 149.5, 127.7 after the fix | sync-introduced, communication-regression |
