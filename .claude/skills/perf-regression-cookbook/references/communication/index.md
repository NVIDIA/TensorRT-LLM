# Regression Cookbook — Communication

This module is the collective layer: the AllReduce implementations and their
strategy selection (`cpp/tensorrt_llm/.../customAllReduce*`,
`tensorrt_llm/_torch/distributed/`), the MNNVL / NVLS / one-shot custom paths,
and the NCCL fallback behind them. It only exists at TP>1 / EP>1, and its
defining property is that a wrong choice is still *correct* — a fast backend
that declines to run hands the work to NCCL and nothing fails. Regressions here
are therefore diagnosed by asking **which implementation actually ran**: read
the backend-selection log lines and the collective kernel names in a trace, and
compare them against the strategy you believe was intended. The second thing to
check is whether the collective is even the cost: the Python-side tactic lookup
and buffer bookkeeping *around* an allreduce can dominate on small models while
the collective kernels are untouched.

## Recurring patterns in this module

- **Fast collective silently fell back** — strategy selection depends on
  constructor args, declared dtype/capability hints and probes; when one goes
  missing or over-rejects, the op runs on NCCL without failing. Fallbacks must
  log loudly, and a capability guard must test the actual requirement rather
  than a proxy. _(Instance: Nemotron-H MNNVL→NCCL fallback. The AutoDeploy
  SYMM_MEM→NCCL sibling was removed on 2026-08-12 — nvbug 6221450 is a
  functional bug, an L0 accuracy assertion, and the slow allreduce was
  spotted while root-causing that. Same mechanism, no perf filing.)_
- **Strategy heuristic keyed on one dimension** — AUTO selection that decides
  from token count alone is reproducibly wrong wherever the real crossover also
  depends on hidden_size, the attached fusion op or the SM generation. Pin each
  strategy explicitly and compare against AUTO: if any explicit choice wins, the
  heuristic is the defect and not the kernels. The fix shape is a measured table
  whose out-of-range default is the portable implementation.
  _(Instance: the AllReduce AUTO strategy LUT.)_
- **Host work on the hot path** — the collective ran fine; the host-side
  bookkeeping around it cost the time, so only small (host-bound) models regress
  and CUDA-graph mode is unaffected. A default flip validated on large dense
  models can hand the bill to eager mode.
  _(Instance: AllReduce host overhead on small-model TP.)_
- **Measurement, not product** — on a shared multi-node cluster the headline is
  often cross-node variance. Pin every probe in the window to ONE node with ≥3
  reps before bisecting: the 9.65% GB300 Llama headline measured 2.42% same-node,
  and the sibling 8B case did not reproduce at all.
  _(Instance: AllReduce host overhead on small-model TP, which is filed under
  both classes for exactly this reason.)_

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [AllReduce host overhead on small-model TP](allreduce-host-overhead-small-model-tp.md) | rc16→rc17 Llama headline −9.65% total token throughput on GB300; 2.42% same-node | host-work-added, measurement-artifact |
| [AllReduce AUTO keyed only on token count](allreduce-strategy-lut.md) | one-shot custom kernel chosen where plain NCCL was faster on A100/H100 | communication-regression, kernel-selection-regression |
| [Nemotron-H allreduce silently falls back from MNNVL to NCCL](nemotronh-mnnvl-nccl-fallback.md) | every allreduce on the NCCL path on NVL multi-node, "with worse perf"; no delta stated | fast-path-fallback, communication-regression |
