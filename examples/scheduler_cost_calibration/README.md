# Scheduler Cost Calibration

Derives the two knobs for **equal-cost context chunking** (KV cache manager V2
scheduler) from the iteration logs of an ordinary run:

- `TLLM_V2_CTX_COST_KV_OFFSET` — kv-token equivalents of the KV-independent
  per-token work; a model × kernel × precision property.
- `TLLM_V2_CTX_COST_KV_DEPTH_THRESHOLD` — the KV depth above which chunks shrink; a
  workload property (a low percentile of the observed depth distribution).

## Workflow

1. Run your deployment with the target workload under the plain token budget
   for ~30 minutes, with per-rank iteration logging on the context workers:

   ```bash
   TLLM_PROFILE_LOG_RANKS=all trtllm-serve ...
   ```

2. Fit both knobs from the worker log(s):

   ```bash
   python fit_context_cost.py /path/to/ctx_worker.log
   ```

   The report includes the fitted cost model, a piecewise-linearity check,
   the measured lockstep waste (share of attention-DP GPU time spent waiting
   for the straggler rank — if this is small, stop here), the predicted
   equal-cost gain, and the ready-to-paste env values.

3. Re-run with the two env vars set on the context workers and compare
   throughput / TTFT / per-rank iteration-time spread against step 1.

Re-fit after changing the model, attention kernels, precision, or GPU
(`kv_cost_offset` moves) or the workload mix (`kv_depth_threshold` moves).
