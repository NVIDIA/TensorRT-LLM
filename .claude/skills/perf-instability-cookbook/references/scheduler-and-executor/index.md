# Instability Cookbook — Scheduler & executor

This module is the Python executor that drives the decode loop and, just as
importantly, the several construction paths that build it: `PyExecutor`
(`tensorrt_llm/_torch/pyexecutor/py_executor.py`) with its
`customized_gc_thresholds` wrapper around the executor loop, and the alternate
AutoDeploy shim `create_autodeploy_executor()`
(`tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py`), which enumerates the
`PyExecutor` kwargs by hand. Instability located here is **host-side**: the
executor process holds an enormous long-lived Python object graph (FX graph +
CUDA-graph wrappers + mamba/SSM caches + MoE state), so any host pause landing
between iterations shows up as ITL spikes and run-to-run throughput spread while
a GPU-only profile looks clean. A second executor construction path is the
recurring source, because a tuning field that is simply not forwarded reverts to
a language default with no error and no warning.

First thing to check — before profiling anything — is the effective
configuration inside the executor process: `gc.get_threshold()` should report the
configured value, not `(700, 10, 10)`. Then look for decode gaps with the device
idle and the host in `gc.collect` (`py-spy dump` during a stall).

## Recurring patterns in this module

- **GC pause on the decode path** — no collective involved: a host-side gen-2
  collection over a huge object graph stalls the decode loop for a fraction of a
  second at unpredictable intervals. TRT-LLM tunes this with
  `garbage_collection_gen0_threshold` (`TorchLlmArgs` default 20000), but the
  field is `Optional[int] = None` and its `None` branch makes
  `customized_gc_thresholds` a silent no-op, so any executor construction path
  that forgets to forward it runs at CPython's default 700 — about 28× more
  frequent — with nothing logged. The general form is broader than GC: **a perf
  tuning knob that defaults to "off" cannot report its own absence.** Prefer a
  required parameter, a shared construction helper, or a startup log line
  stating the effective value, and grep every `PyExecutor(` construction site
  for the field.
  _(Instance: [AutoDeploy never forwarded the gen-0 threshold](autodeploy-gc-gen0-threshold-unforwarded.md).)_
  Distinguish from `pattern-gc-driven-collective-destruction`
  ([DeepEP MoE hang](../moe/moe-deepep-gc-hang.md), now in the MoE module):
  there a finalizer fires a *collective*, so GC skew across ranks hangs peers;
  here a single rank simply pauses, and every peer stays healthy.

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [AutoDeploy never forwarded `garbage_collection_gen0_threshold`](autodeploy-gc-gen0-threshold-unforwarded.md) | gen-2 collections periodically stall decode for 0.5–1 s with the device idle — ~286 vs ~342 tok/s/user (~20 % spread) on Nemotron-3-Nano-30B-A3B-FP8 TP=4, identical config | gc-pause-on-decode-path |
