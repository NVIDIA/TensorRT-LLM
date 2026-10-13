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

## Per-chunk depth term

The two knobs above price a chunk as `(kv_cost_offset + kv_depth) * n`, so
every cost scales with the chunk's token count `n`. Chunked prefill also
re-reads and re-projects the request's cached prefix on every chunk, which
scales with the depth and not with `n`. One optional term (default 0 = off)
charges that in the same budget:

- `TLLM_V2_CTX_COST_PER_CHUNK_DEPTH_TOKENS=R` — each chunk also costs
  `R * kv_depth`, i.e. like `R` extra tokens at the request's depth. `R` is the
  per-chunk prefix cost per cached token divided by the attention cost per
  token*kv, a model x kernel x GPU property like `kv_cost_offset`.

The budget grows by `R * kv_depth_threshold` so a single request at the
threshold still fits a full chunk. `fit_context_cost.py` does not fit this
term (it attributes the lockstep iteration time to a single request, which
folds the per-chunk work into `kv_cost_offset`); per-rank attention timing is
needed to separate them.
