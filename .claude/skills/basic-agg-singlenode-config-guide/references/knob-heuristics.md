# Source-Backed Tuning Notes

Read an exact or nearby checked-in config and the model's deployment guide **before** using these notes. These are not universal thresholds.

## Commonly Tuned Fields

| Field | Guidance |
|---|---|
| `max_batch_size` | Affects throughput. Reachable batch size depends on total sequence length and GPU memory. |
| `max_num_tokens` | Affects throughput and memory. Tune together with `max_batch_size`. |
| `enable_attention_dp` | Scenario-dependent. Follow the exact model guide/config. |
| `kv_cache_config.free_gpu_memory_fraction` | OOM lever. Guides often adjust `max_batch_size` or `max_seq_len` first. |
| `moe_expert_parallel_size` | MoE only. Copy from checked-in source; do not assume it equals TP. |
| `moe_config.backend` | Set only when model guide or checked-in config specifies it. |
| `stream_interval` | Model-specific variation; no single global value. |
| `num_postprocess_workers` | Present in some streaming/higher-concurrency configs. |
| `attention_dp_config.*` | Preserve when present in source configs. |
| `cuda_graph_config.max_batch_size` / `batch_sizes` | Align with source config. Not necessarily equal to server `max_batch_size`. |
| MTP fields (`num_nextn_predict_layers`, `use_relaxed_acceptance_for_thinking`, `relaxed_topk`, `relaxed_delta`) | Only under the **MTP carveout**. Copy verbatim from selected config. |

## Bench-Derived Hints (fallback only)

These come from `tensorrt_llm/bench/` and are benchmark heuristics, not serve rules. Mark any bench-derived value as unverified.

- `throughput.py` and `build.py` seed target ISL/OSL from `DatasetMetadata.avg_isl`/`avg_osl` — use as a fallback when no exact scenario match exists.
- `tuning.py` derives `max_batch_size` and `max_num_tokens` together from ISL/OSL and estimated KV-cache capacity. Rounds upward into coarse buckets; enforces min `max_num_tokens` of 2048. Outputs are "slightly optimistic" upper bounds for benchmark engine builds.
- `general.py` raises `max_num_tokens` to at least `max_isl + max_batch_size` on the non-chunked-prefill path. Benchmark safeguard, not a serve invariant.
- `configuration.py` fills `cuda_graph_config.max_batch_size` from runtime `max_batch_size` when both `batch_sizes` and `max_batch_size` are unset. Skip if the source config already sets graph batch sizes.
- If a bench-derived value diverges from nearby checked-in serve configs, prefer the serve config.

## OOM Triage

Follow the exact model guide/config first. Common levers: lower `max_batch_size`, `max_num_tokens`, `max_seq_len`, or `free_gpu_memory_fraction`. If the deployment guide provides a backend support matrix, use it for `moe_config.backend`.
