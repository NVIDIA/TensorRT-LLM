# Instability Cookbook — MoE

This module is the fused-MoE stack under
`tensorrt_llm/_torch/modules/fused_moe/` — `ConfigurableMoE` and the pluggable
expert-parallel communication backends it selects between (DeepEP,
DeepEPLowLatency, AllGatherReduceScatter), including the fallback path from one
backend to another mid-run. What makes this location interesting for instability
is **resource lifecycle across ranks**: an MoE backend owns distributed buffers
whose release runs a collective, so the timing of Python object destruction
becomes a cross-rank liveness property rather than a local detail. The
observable is a run that either finishes or hangs on identical input, seed and
nodes.

First thing to check on a suspected instance: `py-spy dump` the stuck rank and
look for a `__del__` frame entering a barrier or other collective, then grep the
backend for release paths that are implicit
(`grep -nE 'intranode::barrier|Buffer.__del__|destroy\(\)' tensorrt_llm/_torch/modules/fused_moe/`).

## Recurring patterns in this module

- **GC-driven collective destruction** — a collective op runs inside a
  destructor / finalizer, and Python GC timing, which is not synchronised across
  ranks, decides when the barrier fires; the rank that collects first enters the
  barrier and blocks on peers that have not collected yet. Any collective
  resource whose destructor synchronises MUST have an explicit release path —
  an explicit `destroy()` invoked at a well-defined program point (context
  manager scope, explicit close) rather than left to GC.
  _(Instance: [DeepEP MoE hang on non-deterministic GC](moe-deepep-gc-hang.md).)_
  Distinguish from `pattern-gc-pause-on-decode-path`
  ([AutoDeploy gen-0 threshold](../scheduler-and-executor/autodeploy-gc-gen0-threshold-unforwarded.md),
  now in the scheduler & executor module): there the harm is a host-side *pause*
  in the decode loop on a single rank with no collective involved; here it is a
  collective fired from a finalizer, which hangs peers.

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [DeepEP `Buffer.__del__` calls a collective](moe-deepep-gc-hang.md) | multi-GPU DeepEP / DEEPEPLOWLATENCY on H100 hangs at rank drain — at test end or during the DeepEP → AllGatherReduceScatter fallback; same input, seed and nodes either finish or hang, so CI read it as flaky and skipped the tests | gc-driven-collective-destruction |
