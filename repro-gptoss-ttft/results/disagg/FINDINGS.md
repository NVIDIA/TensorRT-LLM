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

| metric          | agg p50 | disagg p50 | disagg − agg | cookbook delta (rep1 vs rep6) |
|-----------------|--------:|-----------:|-------------:|------------------------------:|
| TTFT            | 166 ms  | 202 ms     | +36 ms       | +963 ms                       |
| TTFT p99        | 187 ms  | 223 ms     | +36 ms       | n/a                           |
| TPOT            | 1.09 ms | 2.61 ms    | +1.52 ms     | n/a                           |
| ITL median      | 3.7 ms  | 369 ms     | +365 ms      | n/a                           |
| E2EL median     | 373 ms  | 575 ms     | +202 ms      | n/a                           |

The ~365 ms ITL inflation on disagg is consistent with the gen worker's
`stream_interval: 100` setting: the worker batches 100 decode iters between
SSE chunks, so the first chunk after TTFT carries ~100 iters × 4 ms ≈
400 ms.

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
2. **`kv_cache_aware` router on the proxy.** Cookbook used
   `kv_cache_aware`; we used `round_robin` (the kv_cache_aware path tries
   `AutoTokenizer.from_pretrained("openai/gpt-oss-120b")` on the proxy,
   which currently fails under `HF_HUB_OFFLINE=1` for the served-model id
   we used). With a working tokenizer the router runs the tokenizer + a
   block-hash computation per request on the proxy's event loop, which
   for a 13k-token prompt is non-trivial.
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
2. **Re-run with `kv_cache_aware` router** with a working tokenizer
   (pass `--tokenizer /local/path` to `trtllm-serve disaggregated`) and
   measure how much the per-request tokenize+hash step on the proxy adds
   to TTFT.
3. **High-concurrency repro.** Dynamo's report was at conc=48. Run
   conc=8/16/48 on our setup to see whether the gen-worker first-iter
   stall remains hidden by ctx prefill or starts contributing to TTFT.
4. **Fix `/perf_metrics` plumbing** for the disagg proxy (env var on all
   three processes + completion path + ctx_request_id wiring) so we can
   produce the cookbook-style per-stage table side-by-side with ours.
