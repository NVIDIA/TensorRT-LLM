# Independent RWLT analysis (agent workload)

Methodology: drive the same `trtllm-serve` agg and disagg setups used in
[results/agg/FINDINGS.md](../results/agg/FINDINGS.md) and
[results/disagg/FINDINGS.md](../results/disagg/FINDINGS.md) with the RWLT
client (`python3 -m rwlt.run`) at conc=1 against the multi-turn
agentic-coding trajectories dataset:

- Dataset: `/home/scratch.shobhitv_coreai/aa-rwlt-shared-datasets/aa-rwlt_coding-agent-scenario_tuning_v2.jsonl`
- RWLT config: [`configs/rwlt_baseline.yaml`](../configs/rwlt_baseline.yaml) (conc=1, seed=42, `min_total_trajectories: 30`)
- Server configs: [`configs/repro_agg_tp1_eagle3.yaml`](../configs/repro_agg_tp1_eagle3.yaml),
  [`configs/repro_disagg_ctx_tp1.yaml`](../configs/repro_disagg_ctx_tp1.yaml),
  [`configs/repro_disagg_gen_tp1.yaml`](../configs/repro_disagg_gen_tp1.yaml),
  [`configs/repro_disagg_proxy.yaml`](../configs/repro_disagg_proxy.yaml) (kv_cache_aware router)
- Run wrapper: [`scripts/run_session.sh`](../scripts/run_session.sh) (auto-teardown after every session)
- Cross-run comparison: [`scripts/diff_rwlt_runs.py`](../scripts/diff_rwlt_runs.py)

Both runs deterministically selected the same set of 30 trajectories from
seed=42, producing 1172 measured turn-requests in each. All matched 1:1 on
`(conversation_id, conversation_idx)`.

## Bench-level result

| metric           | agg p50 | agg p99 | disagg p50 | disagg p99 | disagg − agg p50 |
|------------------|--------:|--------:|-----------:|-----------:|-----------------:|
| TTFT             | 88 ms   | 329 ms  | 451 ms     | 993 ms     | **+346 ms**      |
| avg input/req    | ~40k tokens | — | ~40k tokens | — | — |
| cache hit rate (server) | ~95% | — | ~95% | — | — |

(Cache hit rate is realistic for multi-turn agent workloads — each turn's
input includes the full prior chat history that the previous turn already
prefilled.)

The cookbook report rep6 (single request, 13,576-token prompt) measured
+963 ms agg→disagg, with `prefill_finalize 767 ms + prefill→decode 986 ms`
as the dominant buckets. Our random-IDs benchmark_serving run measured
+36 ms at the same conc=1 with a 13,576-token prompt.

| comparison | workload | delta TTFT p50 |
|---|---|---:|
| random-IDs benchmark_serving (this repro) | 1 prompt × 13576 raw token IDs | +36 ms |
| RWLT agent workload (this repro) | 30 trajectories × ~40k template+history tokens | +346 ms |
| RWLT-equivalent cookbook (single request, KV-reuse warmup) | 1 trajectory turn × 13576 tokens | +963 ms |

## Per-turn observation: delta grows with input length

Bucketing the 1172 matched turns by ISL (data from
`rwlt-results/diff_agg_vs_disagg.tsv`):

| ISL bucket   | n   | median delta ms | μs/token |
|--------------|----:|----------------:|---------:|
| 9k – 14k     | sample | ~150 | ~12 |
| 14k – 20k    | sample | ~210 | ~12 |
| 20k – 30k    | sample | ~400 | ~16 |
| 30k – 40k    | sample | ~550 | ~16 |
| 40k+         | sample | ~620 | ~15 |

(Bucket medians from the per-turn TSV; rerun
`scripts/diff_rwlt_runs.py --per-turn-out ... ` to refresh.)

Cross-check with the random-IDs run: 13,576 tokens × 2.6 μs/token = 36 ms,
which matches the +36 ms benchmark_serving number exactly. RWLT shows
12–20 μs/token, ~5–8× more per-token disagg overhead.

The most likely contributors to that per-token scaling, in order of
expected magnitude:

1. **`kv_cache_aware` router on the disagg proxy** runs
   `AutoTokenizer.apply_chat_template` + block-hash on the proxy event
   loop for every request. For 40k tokens that is non-trivial CPU work
   blocking the asyncio loop. The random-IDs run did not exercise this
   path because the bench client sent raw token IDs.
2. **Chat-template re-tokenization on ctx and gen workers.** Each worker
   independently tokenizes the full request body before scheduling. Two
   tokenizations × 40k tokens ≈ tens of ms.
3. **Request body marshalling.** Serializing/deserializing a 40k-token
   `messages` array through the proxy adds bandwidth-bounded latency.

## Cold vs warm prefix has the same delta

`diff_rwlt_runs.py` split by `cached_tokens / input_tokens ≥ 0.5`:

| population | n | median delta ms |
|---|---:|---:|
| cold-prefix turns (cache_ratio < 0.5) | 46 | 333.6 |
| warm-prefix turns (cache_ratio ≥ 0.5) | 1126 | 346.7 |

This is consistent with the per-token scaling above being dominated by
work that doesn't care about cache state: tokenization and per-request
proxy bookkeeping. KV transfer time itself (which should scale only with
*new* tokens) is small relative to those.

## Per-conversation outliers

Top 10 conversations by |median TTFT delta| (the conversations with
deepest turn counts and longest contexts):

| conversation_id | turns | median delta ms |
|---|---:|---:|
| aa-rwlt-coding-agent-043 | 1 | 805 |
| aa-rwlt-coding-agent-056 | 61 | 592 |
| aa-rwlt-coding-agent-010 | 79 | 515 |
| aa-rwlt-coding-agent-011 | 63 | 415 |
| aa-rwlt-coding-agent-051 | 47 | 412 |
| aa-rwlt-coding-agent-016 | 65 | 405 |
| aa-rwlt-coding-agent-027 | 52 | 389 |
| aa-rwlt-coding-agent-093 | 42 | 386 |
| aa-rwlt-coding-agent-002 | 62 | 385 |
| aa-rwlt-coding-agent-050 | 21 | 370 |

These are all the trajectories that accumulate large chat histories, so
each turn's input grows with turn index, matching the per-token scaling
above.

## Note on ISL agreement

`diff_rwlt_runs.py` reports 533 / 1172 turns with exactly matching ISL.
The other 639 turns show **disagg ISL = agg ISL − 2** (almost always
exactly 2 tokens shorter on the disagg side). The pattern starts on
turn 0 of several conversations (e.g. `aa-rwlt-coding-agent-002` turn 0:
agg=9529, disagg=9527), where the input is byte-identical between the
two layouts — so the difference is server-side template handling, not
request payload. Two tokens out of 10k–80k input is functionally noise
for TTFT comparison, but worth noting because it suggests a small
template-handling divergence between the OpenAI handler in IFB vs disagg
mode in the current TRT-LLM build.

## Output token / spec acceptance variance

Per-turn output token delta (disagg − agg): p50 = 0, mean = −10, with
single-conversation extremes of ±16k tokens. Eagle3 spec acceptance
sampling is RNG-state dependent per executor instance, so once one turn
produces a different assistant message the rest of that conversation
diverges. Doesn't affect per-turn TTFT comparison (we're comparing
matched turns 1:1), but worth noting if you want to compare end-to-end
conversation completion times.

## Next ablation: tease out where the per-token cost lives

Three knobs to change at once (matches the in-tree gpt-oss-120b convention
documented in [COOKBOOK_VS_INREPO_CONFIG_DIFF.md](../COOKBOOK_VS_INREPO_CONFIG_DIFF.md)
and isolates spec-decoding effects from the rest):

1. `num_postprocess_workers: 4` on all workers (currently 4 on agg + ctx,
   0 on gen — the cookbook gen-worker value)
2. `stream_interval: 20` on all workers (currently default 1 on agg,
   10 on ctx, 100 on gen — the in-tree gpt-oss-120b value)
3. Drop `speculative_config` entirely on all workers (isolates Eagle3 cost)

Pre-built ablation YAMLs:
[`configs/repro_agg_tp1_tuned.yaml`](../configs/repro_agg_tp1_tuned.yaml),
[`configs/repro_disagg_ctx_tp1_tuned.yaml`](../configs/repro_disagg_ctx_tp1_tuned.yaml),
[`configs/repro_disagg_gen_tp1_tuned.yaml`](../configs/repro_disagg_gen_tp1_tuned.yaml).

Drive via `scripts/run_session.sh` once those are created.
