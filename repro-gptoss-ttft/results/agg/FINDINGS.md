# Independent agg analysis

Methodology: parse the `print_iter_log: true` server output (no extra code,
no nsys), grouped per request by ctx-iter boundaries. See
[scripts/parse_iter_log.py](../../scripts/parse_iter_log.py).

Run: 32 random-IDs prompts at conc=1 against `trtllm-serve` agg TP1 with
the cookbook Eagle3-v3 (Eagle3-next) config. The first request is dropped
because `benchmark_serving.py`'s "Initial test run" sends the same prompt
as warmup and the measured first request lands as a near-complete KV-cache
hit (with `enable_block_reuse: true`). 31 fresh-prefill requests remain.

## Per-iter timing distribution (31 requests, 31 ctx iters, 2876 gen iters)

| bucket | min ms | p50 ms | mean ms | p90 ms | p99 ms | max ms |
|---|---:|---:|---:|---:|---:|---:|
| **ctx iter host_step_time** (CPU launch+sched) | 54 | 68 | 73 | 72 | 180 | 180 |
| **ctx GPU work** (prev_device of first-gen iter) | 159 | 175 | 173 | 180 | 184 | 185 |
| gen first-after-ctx host_step_time | 2 | 110 | 103 | 112 | 113 | 113 |
| gen first-after-ctx prev_device_step_time | 159 | 175 | 173 | 180 | 184 | 185 |
| gen steady-state host_step_time | 1.5 | 4.0 | 4.0 | 4.3 | 4.5 | 11 |
| gen steady-state prev_device_step_time | 3.8 | 4.1 | 4.2 | 4.3 | 4.4 | 7 |
| num_ctx_tokens per request | 13687 | 14002 | 13970 | 14031 | 14040 | 14041 |

## Observations from the iter log alone

1. **Ctx compute on B200 TP1 + Eagle3-v3 is ≈175 ms p50** for a 14k-token
   prompt; p50→p99 spread of 175→184 ms (5% variance).
2. **The first decode iter after ctx costs ~110 ms host time, vs 4 ms
   steady-state — a 25× spike.** Possible contributors (not yet isolated):
   - Eagle3 draft heads first-call setup / draft tokens initialization
   - First-token postprocess + IPC handoff (kept inline by `num_postprocess_workers=4`)
   - CUDA-graph re-capture for a transition path not covered by warmup
   - Scheduler bookkeeping on the ctx→gen transition
3. **Steady-state decode is 4 ms per 4-token Eagle3 iter ≈ 1 ms/output
   token.** Matches the bench-side TPOT = 1.09 ms. Eagle3-v3 + cuda graph +
   spec decoding is producing the expected acceleration.
4. **CTX host_step_time = 68 ms is decoupled from CTX GPU time (175 ms)**
   because the async pipeline overlaps host scheduling/launch with the
   prior gen iter's GPU work. The CTX iter's `prev_device_step_time` is
   ~4 ms = previous gen iter, not the ctx GPU time. The ctx GPU time
   shows up as `prev_device_step_time` of the first GEN iter (~175 ms).

## TTFT decomposition from /perf_metrics

Per-request perf_metrics aggregate (31 requests, warmup dropped):

| stage | p50 ms | what's in here |
|---|---:|---|
| server_arrival → arrival | 27 | request parsing + tokenization (server-side) |
| arrival → first_scheduled | 4 | executor queue wait |
| first_scheduled → first_token | 134 | ctx scheduling + ctx GPU compute (overlapped) |
| first_token → server_first_token | 1.4 | postproc + IPC to OpenAI server |
| **TTFT total** | **~166** | |

The 134 ms `first_scheduled → first_token` is consistent with ctx GPU ≈
175 ms minus ~40 ms of pipeline overlap with the prior ctx-iter launch.

## Side-by-side with the cookbook agg server lane (rep1, single request)

| stage | this repro (p50, 31 reqs) | cookbook (rep1, single req) |
|---|---:|---:|
| server ingress | 27 ms | 170 ms |
| queue | 4 ms | 2 ms |
| context/prefill + first token | 134 ms | 41 ms |
| executor → server token | 1.4 ms | 737 ms |
| **TTFT total** | **166 ms** | **950 ms** |

Two structural differences worth flagging:

- **Cookbook ctx+first-token at 41 ms** comes from a single-request run where
  the same prompt was sent twice (warmup + measured) with
  `enable_block_reuse: true`, so the measured ctx phase saw a near-complete
  KV reuse. Our 134 ms is the value for a fresh-prefill request.
- **Cookbook `executor → server token` at 737 ms** vs ours at 1.4 ms p50 /
  12 ms p99: this bucket is the gap between
  `request_perf_metrics.timing_metrics.first_token_time` and the OpenAI
  server's `raw_request.state.server_first_token_time`. Both are
  in-process timestamps; in this repro it consistently lands at 1–2 ms.

## What this doesn't yet explain

- The 110 ms "first decode after ctx" host-time spike — needs a Python-level
  trace inside `_handle_responses` / `_executor_loop` / the spec drafter init
  to attribute precisely. Becomes important when comparing against the
  disagg `prefill_finalize` bucket.
- KV transceiver behaviour: not exercised by agg.
- Effect of `TRTLLM_ENABLE_PDL=1` (still off in this run): unknown without
  a re-run.

## Next steps

1. Re-run with `TRTLLM_ENABLE_PDL=1` env on the server. Hypothesis: ctx GPU
   drops 5–10%; first-after-ctx host spike unaffected.
2. Run the disagg trio and apply the same iter-log parsing to ctx + gen logs
   independently, plus the per-request perf_metrics aggregator. Map any
   `prefill_finalize`-like gap to whether it shows up on the ctx-server iter
   log or only in the `server_first_token_time` bookkeeping on the proxy.
