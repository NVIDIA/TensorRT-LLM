# Source-Backed Tuning Notes

Use these notes only **after** reading an exact or nearby checked-in config and the model's current deployment guide. These are not universal thresholds.

Preserve the user's explicit performance objective before borrowing knob values from any nearby config. If the nearby config is unlabeled for latency / balanced / throughput intent, treat it as a default starting point rather than a verified objective match.

## Commonly Tuned Fields in Current Repo Sources

| Field | What current checked-in docs/configs support |
|---|---|
| `max_batch_size` | Affects throughput and should not be set too low. Current docs note the reachable batch size depends on total sequence length and available GPU memory. |
| `max_num_tokens` | Affects throughput and memory pressure. Tune it together with `max_batch_size`. |
| `enable_attention_dp` | Current checked-in configs show it is scenario-dependent. Keep or change it based on the exact model guide/config you are following, not on a global threshold. |
| `kv_cache_config.free_gpu_memory_fraction` | One possible OOM lever. Current checked-in guides often adjust `max_batch_size` or `max_seq_len` first, depending on the model. |
| `moe_expert_parallel_size` | For MoE models, copy the exact or nearest checked-in value for the same model family and verify it against the model-specific guide/config you are following. Do not assume it always equals TP. |
| `moe_config.backend` | Only set this when the model-specific deployment guide or a checked-in config calls for it. |
| `stream_interval` | Current configs show model-specific variation; do not assume one global value. |
| `num_postprocess_workers` | Present in some current streaming / higher-concurrency configs; keep or remove it based on nearby checked-in configs. |
| `attention_dp_config.*` | Preserve it when present in nearby source-backed configs. |
| `cuda_graph_config.max_batch_size` / `batch_sizes` | Keep it aligned with the batching strategy shown in current docs/configs. |

## Narrow Bench-Derived Hints

These come from `tensorrt_llm/bench/` and are benchmark / engine-build heuristics, not universal `trtllm-serve` rules. Only borrow them intentionally as a fallback, and mark the result as unverified.

- Bench-only fallback: use dataset averages as first local targets when no exact scenario match exists. `tensorrt_llm/bench/benchmark/throughput.py` and `tensorrt_llm/bench/build/build.py` seed `target_input_len` / `target_output_len` from `DatasetMetadata.avg_isl` / `avg_osl`.
- Treat `max_batch_size` and `max_num_tokens` as coupled knobs. `tensorrt_llm/bench/build/tuning.py` derives them together from target ISL/OSL and estimated KV-cache capacity rather than tuning them independently.
- Treat too-small `max_num_tokens` as a warning sign on the current bench non-`enable_chunked_prefill` path. `tensorrt_llm/bench/benchmark/utils/general.py` raises `max_num_tokens` to at least `max_isl + max_batch_size` in that path. Use this as a benchmark safeguard, not a universal serve invariant.
- `tensorrt_llm/bench/build/tuning.py` rounds `max_batch_size` and `max_num_tokens` upward into coarse buckets and enforces a benchmark-oriented minimum `max_num_tokens` of `2048`. Treat that as benchmark convenience, not as a general serving default.
- If `cuda_graph_config` leaves both `batch_sizes` and `max_batch_size` unset, `tensorrt_llm/bench/dataclasses/configuration.py` fills `cuda_graph_config.max_batch_size` from runtime `max_batch_size`. Do not apply this if the checked-in config already sets graph batch sizes explicitly.
- `tensorrt_llm/bench/build/tuning.py` describes its outputs as "slightly optimistic" upper bounds for benchmark engine build. If a bench-derived value diverges materially from nearby checked-in serve configs, prefer the checked-in serve config unless you are explicitly building a benchmark-only engine.

## Safe Adjustment Order

1. Lock the user's performance objective (`Min Latency`, `Balanced`, `Max Throughput`, or unspecified)
2. Exact in-scope database config
3. Same-model in-scope database config for a nearby scenario with matching objective when current sources label it
4. Same-model curated config that stays in scope and matches the stated objective when the repo labels it explicitly
5. Model-specific deployment guide / README
6. Small local adjustments, explicitly marked as unverified

## Things to Avoid

- Numeric crossover thresholds that are not stated in the current checked-in guide for that model
- "Only N knobs matter" framing
- Crossing from a latency-oriented config to a throughput-oriented config without calling out the mismatch
- Copying knob values across unrelated model families
- Treating an unlabeled default config as a verified latency / balanced / throughput profile
- Dropping model-specific fields because they look auxiliary
- Assuming a field is constant just because it was constant for one other model

## OOM / Mismatch Triage

- If you hit OOM, follow the exact model guide/config first. Common levers in current checked-in docs/configs include lowering `max_batch_size`, `max_num_tokens`, `max_seq_len`, or `kv_cache_config.free_gpu_memory_fraction`.
- If a nearby checked-in config contains fields your draft omits, preserve them unless the current docs say they are out of scope.
- If the deployment guide provides a backend support matrix, use it instead of guessing `moe_config.backend`.
