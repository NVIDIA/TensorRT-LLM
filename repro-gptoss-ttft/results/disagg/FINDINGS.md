# Independent disagg analysis

Methodology: same as agg — parse `print_iter_log: true` output from the ctx
worker, gen worker, and proxy. 32 random-IDs prompts at conc=1 through the
disagg proxy. Bench top-level via `benchmark_serving.py` against
`/v1/completions`.

> Note: `/perf_metrics` fetch via the proxy returned empty in this run. The
> disagg proxy joins ctx+gen perf metrics by `ctx_request_id` (see
> `tensorrt_llm/serve/perf_metrics.py:DisaggPerfMetricsCollector`), which
> requires the env var on all three processes and request-level wiring we
> didn't set up correctly when manually restarting the proxy. The iter-log
> analysis below is independent of that endpoint.

## Bench-level result

All TTFT/TPOT/ITL/E2EL numbers below are at conc=1, 32 random-IDs prompts,
ISL=13576 / OSL=144, cold ctx KV cache (see "KV cache reuse note" below).

| metric       | agg p50 (seed=42) | disagg round_robin p50 (seed=42) | disagg kv_aware p50 (seed=43) | round_robin − agg | kv_aware − round_robin | cookbook rep1 → rep6 |
|--------------|------------------:|---------------------------------:|------------------------------:|------------------:|-----------------------:|---------------------:|
| TTFT         | 166 ms            | 202 ms                           | 202 ms                        | +36 ms            | ~0 ms                  | +963 ms              |
| TTFT p99     | 187 ms            | 223 ms                           | 207 ms                        | +36 ms            | −16 ms                 | n/a                  |
| TPOT         | 1.09 ms           | 2.61 ms                          | 2.40 ms                       | +1.52 ms          | −0.2 ms                | n/a                  |
| ITL median   | 3.7 ms            | 369 ms                           | 341 ms                        | +365 ms           | −28 ms                 | n/a                  |
| E2EL median  | 373 ms            | 575 ms                           | 547 ms                        | +202 ms           | −28 ms                 | n/a                  |

The ~365 ms ITL inflation on disagg is consistent with the gen worker's
`stream_interval: 100` setting: the worker batches 100 decode iters between
SSE chunks, so the first chunk after TTFT carries ~100 iters × 4 ms ≈
400 ms.

**Router observation.** At conc=1 with 1 ctx + 1 gen server, switching
from `round_robin` to `kv_cache_aware` has no measurable effect on TTFT.
With a single ctx server the router has no real routing decision to make,
so the per-request `AutoTokenizer.apply_chat_template` + block-hash work
on the proxy thread (~13k tokens) is either fast enough to be lost in the
TTFT noise floor, or fully hidden by the network/dispatch path. The minor
TPOT/ITL/E2EL improvements likely reflect spec acceptance variance across
different prompt sets rather than router behaviour.

## KV cache reuse note (important when comparing runs)

`benchmark_serving.py --random-ids --seed N` generates 32 prompts
deterministically from N. Same seed → identical token IDs across runs.
With `enable_block_reuse: true` on the ctx worker, running two benches
in a row against the same server with the same seed makes the second run
hit the KV cache on every prompt, which lowers the reported TTFT to
near-zero prefill cost.

Concrete example from this experiment: the first kv_aware run (with the
same seed=42 as the prior round_robin run, against still-warm workers)
reported TTFT p50 = 76 ms. The ctx iter log showed `num_ctx_tokens = 1`
on every iter — i.e. the entire 14k-token prefill was served from the
KV pool the round_robin run had populated. We re-ran with seed=43 to get
an uncached measurement (202 ms shown above).

Rule of thumb when chaining benches:

| run | TTFT meaning |
|-----|--------------|
| first run after worker launch (any seed) | cold ctx, full prefill |
| later run with a **new seed** | cold ctx, full prefill |
| later run with the **same seed** against still-warm workers | warm ctx, mostly KV reuse |

Defaults: `scripts/bench_ttft.sh` uses `SEED=42` unless overridden. For
back-to-back runs without restarting workers, pass `SEED=43`, `SEED=44`,
... per run.

## What the iter logs show (ctx + gen, independent of /perf_metrics)

### CTX worker (logs/ctx.log, 63 real prefill iters after warmup)

| bucket | min ms | p50 ms | mean ms | p90 ms | p99 ms | max ms |
|---|---:|---:|---:|---:|---:|---:|
| host_step_time (big prefill ≥10k toks) | 16 | **131** | 287 | 510 | 551 | 560 |
| prev_device_step_time (≈ ctx GPU work) | 1 | **215** | 291 | 637 | 679 | 688 |
| num_ctx_tokens | 13687 | 14002 | 14066 | 14036 | 14041 | 20000 |

CTX GPU work p50 = **215 ms** on the disagg ctx worker vs 175 ms on the
agg server. 40 ms more — candidates include chunked-prefill behaviour
(63 prefill iters for ~32 requests implies ~2 ctx iters per request on
average) or Eagle3 spec on ctx (`decoding_type: Eagle` is set in
`repro_disagg_ctx_tp1.yaml`).

### GEN worker (logs/gen.log, 3136 gen iters across 32 requests)

| bucket | min ms | p50 ms | mean ms | p90 ms | p99 ms | max ms |
|---|---:|---:|---:|---:|---:|---:|
| host_step_time (steady-state gen) | 1.4 | 4.0 | 6.0 | 4.2 | 86.9 | 229 |
| prev_device_step_time (steady-state) | 0.5 | 4.1 | 6.1 | 4.3 | 86.9 | 229 |
| **first iter of each request** (n=31) | 186 | **201** | 200 | 204 | 229 | 229 |
| prev_device at that first iter | 0.5 | 4.0 | 2.8 | 4.2 | 4.3 | 4.3 |

Steady-state gen is 4 ms/iter — identical to agg (Eagle3 working as
expected).

**The 31 "first iter of each request" entries are 201 ms p50 of host time
with the GPU near-idle (prev_dev = 1–4 ms).** This is the gen worker
waiting for the KV cache transfer to arrive from the ctx worker plus the
first decode setup for a newly-arrived request. Per request, the gen
worker carries ~200 ms of host-side wait that the agg path doesn't pay.

### How the +36 ms net TTFT delta arises

Disagg pipeline (mostly overlapped):

```
T = 0       client -> proxy -> ctx worker             (a few ms)
T ≈ 215     ctx worker finishes prefill GPU work      (CTX worker)
T ≈ 215+ε   KV cache transferred to gen worker        (NIXL/UCX, single-digit ms)
T ≈ 215+ε+δ gen worker first decode iter completes    (GEN worker, 4 ms GPU)
T ≈ 215+ε+δ first token streamed back to client       (SSE)
```

The gen worker's 200 ms "first iter host time" is mostly waiting for the
KV transfer to arrive while the ctx worker is still prefilling. Net
overhead added to TTFT after ctx prefill finishes is roughly the
single-digit ms of NIXL transport plus first-decode setup ≈ 25–35 ms,
which matches the observed +36 ms p50 delta.

That ~200 ms gen-worker first-iter cost would become more visible at
higher concurrency where multiple requests' KV transfers can't all hide
inside one ctx prefill; for **conc=1** it is largely hidden.

## How this compares with the cookbook disagg breakdown

Side-by-side of stage-level p50s (cookbook rep6 single request; ours
aggregated over 31 requests):

| stage | this repro (p50) | cookbook (rep6) |
|---|---:|---:|
| TTFT total | 202 ms | 2629 ms |
| ctx context compute | ≈215 ms GPU | 27 ms |
| ctx → gen handoff window | ≈25 ms (NIXL + first decode setup) | 986 ms (`prefill -> decode`) |
| disagg-side `prefill_finalize` | not directly observed | 767 ms |
| KV transfer | single-digit ms (from gen iter pattern) | 78 ms |
| decode compute | 4 ms / iter | 6 ms / iter |

Possible reasons for the difference (in roughly the order most worth
investigating):

1. **Cross-process clock alignment.** Cookbook `prefill_finalize`
   (`ctx_first_token_time → ctx_server_first_token_time`) and
   `prefill → decode` (`ctx_server_first_token_time → gen_server_arrival_time`)
   are derived from timestamps recorded by three separate processes (ctx
   server, gen server, RWLT client) with steady-clock alignment in their
   measurement framework. Any skew or polling pause that lands inside
   those windows shows up in those buckets.
2. ~~**`kv_cache_aware` router on the proxy.**~~ **Tested.** Cookbook
   used `kv_cache_aware`; our first disagg run used `round_robin`. With
   the router switched to `kv_cache_aware` (after pointing the request
   `model` field at the local checkpoint so the proxy tokenizer load
   resolves from disk under `HF_HUB_OFFLINE=1`), TTFT p50 stays at
   202 ms — same as round_robin. With one ctx server the per-request
   `apply_chat_template` + block-hash work on the proxy thread either
   completes within the TTFT noise floor or is hidden by the network /
   dispatch path. Probably not the cookbook's source.
3. **`torch_compile_config` + `enable_piecewise_cuda_graph: true`** with
   the 157-entry `capture_num_tokens` list. Present in both setups; we
   haven't independently isolated whether it adds per-request graph
   validation cost in steady state.
4. **`TRTLLM_ENABLE_PDL` env not set** on the cookbook ctx server. Affects
   ctx forward fusion / kernel overlap. Would shift the ctx GPU bucket,
   not the prefill-finalize bucket.

## Other observations

- **TPOT 2.6 ms on disagg vs 1.1 ms on agg.** Bench-side spec acceptance
  rate on gen is 1.58 tok/iter vs Eagle3's max of 4 — i.e. 1.58/4 = 40%
  of spec tokens accepted. Could be the prompt mix (random IDs aren't
  language-like) rather than a configuration issue.
- **Decode iter count per request ≈ 98 on gen vs ≈ 93 on agg.** Same
  144-token target output, so spec acceptance ratio differs slightly
  between paths.

## Next steps

1. **Confirm the +36 ms delta is stable** by re-running disagg with the
   same configs but `cuda_graph_config: null` on ctx and no
   `torch_compile_config` (matches the perf-sanity disagg pattern). If
   delta stays at +36 ms, current config is well-understood at conc=1.
2. ~~Re-run with `kv_cache_aware` router~~ — done; no measurable TTFT
   change vs round_robin at this topology.
3. **High-concurrency repro.** Dynamo's report was at conc=48. Run
   conc=8/16/48 on our setup to see whether the gen-worker first-iter
   stall remains hidden by ctx prefill or starts contributing to TTFT.
4. **Fix `/perf_metrics` plumbing** for the disagg proxy (env var on all
   three processes + completion path + ctx_request_id wiring) so we can
   produce the cookbook-style per-stage table side-by-side with ours.
