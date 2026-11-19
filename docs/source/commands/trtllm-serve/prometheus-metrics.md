<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prometheus performance metrics

Enable collection in the serving YAML:

```yaml
return_perf_metrics: true
enable_iter_perf_stats: true
enable_iter_req_stats: false
iter_perf_stats_interval: 1
```

```bash
trtllm-serve /path/to/model --config metrics.yaml --port 8000
curl --fail http://localhost:8000/prometheus/metrics
```

Scrape each worker's `/prometheus/metrics` endpoint. OpenEngine defaults to a
separate HTTP metrics listener on port 8001; see its
[metrics configuration](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/grpc/openengine/README.md#prometheus-metrics).
Keep prefill/decode targets and worker `instance` labels separate.

Request histograms accept bucket overrides through `prometheus_metrics_config`;
see the [serving configuration reference](index.rst). Values are seconds and must
be finite, nonempty and strictly increasing. `perf_metrics_output_dir` also
enables collection. `print_iter_log` is independent and can remain disabled.

## Create a Grafana dashboard

Copy and edit `examples/serve/grafana/example.config.json` for your deployment,
then render and preview the dashboard:

```bash
python3 examples/serve/grafana/dashboard.py render --config /tmp/dashboard.json --output /tmp/dashboard
python3 examples/serve/grafana/dashboard.py deploy --config /tmp/dashboard.json --output /tmp/dashboard --dry-run
```

Rendering is offline; remove `--dry-run` to publish. See the
[Grafana example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/serve/grafana)
for datasource/workload selection, API versus ConfigMap ownership, publication
and recovery. Select one scrape path per engine to avoid duplicated observations.
Optional panels remain hidden until native measurements arrive; dashboard setup
does not enable instrumentation. See [tiered KV metrics](#tiered-kv-cache)
for occupancy, movement, sampled migration timing and conversation counts.

## Metric reference

### Request latency, outcomes, and tokens

| Metric | Type | Meaning |
| --- | --- | --- |
| `e2e_request_latency_seconds` | Histogram | Arrival to last output token. |
| `time_to_first_token_seconds` | Histogram | Arrival to first output token. |
| `time_per_output_token_seconds` | Histogram | First-to-last-token duration divided by output tokens minus one; requires at least two output tokens. |
| `request_queue_time_seconds` | Histogram | Arrival to first scheduling; zero is valid for immediate scheduling. |
| `request_prefill_time_seconds` | Histogram | First scheduling to first output token. |
| `request_decode_time_seconds` | Histogram | First to last output token. |
| `request_inference_time_seconds` | Histogram | First scheduling to last output token. |
| `request_success_total` | Counter | Completed output candidates by `finished_reason`; multiple outputs can contribute more than one completion per request. |
| `request_error_total` | Counter | HTTP request errors by `http_code`; use OpenEngine RPC outcomes for gRPC failures. |
| `kv_cache_transfer_duration_seconds` | Histogram | Completed-request KV-transfer duration from existing transfer start/end timestamps; measured in seconds. |
| `kv_cache_transfer_bytes_total` | Counter | Positive KV-transfer bytes reported by completed-request performance metrics. |
| `prompt_tokens_total` | Counter | Input tokens recorded by request performance collection. |
| `generation_tokens_total` | Counter | Output tokens recorded by request performance collection. |
| `prompt_cached_tokens_total` | Counter | Prompt tokens served from KV cache, recorded at request completion. |
| `prompt_cached_tokens_per_request` | Histogram | Cached prompt tokens per recorded request sample, including zero. |
| `prefill_perplexity` | Histogram | Finite prefill perplexity values, when supplied. |
| `generation_perplexity` | Histogram | Finite generation perplexity values, when supplied. |

### Executor-loop diagnostics

| Metric | Type | Meaning |
| --- | --- | --- |
| `loop_scheduled_requests` | Gauge | Scheduled request count at the observed loop boundary. |
| `loop_active_requests` | Gauge | Active requests at that boundary. |
| `loop_unscheduled_requests` | Gauge | Unscheduled request count, preserving the scheduling sample's existing timing scope. |
| `loop_dummy_requests` | Gauge | Dummy request count at that boundary. |
| `loop_adp_dummy_context_tokens` | Gauge | Context dummy tokens captured before the loop counters reset. |
| `loop_adp_dummy_generation_tokens` | Gauge | Generation dummy tokens captured before the reset. |
| `loop_requests_fetched_total` | Counter | Requests assigned to that producing rank, from cumulative snapshots. Repeated loop snapshots do not double count; skipped snapshots retain the cumulative count. |

### Queue load, batching, and limits

| Metric | Type | Meaning |
| --- | --- | --- |
| `num_requests_running` | Gauge | Active requests reported by rank 0. |
| `num_requests_waiting` | Gauge | Queued requests reported by rank 0. |
| `num_requests_completed_total` | Counter | Engine-reported request completions accumulated from iteration records. |
| `max_num_active_requests` | Gauge | Maximum active-request capacity reported by the engine. |
| `max_batch_size_static` | Gauge | Static maximum batch size. |
| `max_batch_size_runtime` | Gauge | Runtime maximum batch size. |
| `max_num_tokens_runtime` | Gauge | Runtime maximum token budget. |
| `num_context_requests` | Gauge | Context/prefill requests in the local batch, by `dp_rank`. |
| `num_generation_requests` | Gauge | Generation/decode requests in the local batch, by `dp_rank`. |
| `num_paused_requests` | Gauge | Paused requests, by `dp_rank`. |
| `num_scheduled_requests` | Gauge | Scheduled requests, by `dp_rank`. |
| `context_kv_tokens` | Gauge | Existing KV-context tokens read by scheduled prefill requests, by rank. |
| `generation_kv_tokens` | Gauge | Summed KV-context length of scheduled generation requests, by rank. |
| `paused_kv_tokens` | Gauge | Summed KV-context length of paused requests, by rank. |
| `total_context_tokens` | Gauge | Context tokens in the latest local batch; despite its name, this is not a counter. |
| `avg_decoded_tokens_per_iter` | Gauge | Engine-reported average decoded tokens per iteration. |
| `prefill_batch_occupancy` | Gauge | Local context-request count divided by reported maximum active-request capacity. |
| `prefill_batch_tokens` | Histogram | Positive context-token counts in exported local iteration records. |

### KV-cache capacity, reuse, and transfers

| Metric | Type | Meaning |
| --- | --- | --- |
| `kv_cache_transfer_requests_total` | Counter | Whole-request transfer outcomes by `dp_rank` and `outcome`; cumulative across sampled iterations. |
| `kv_cache_allocated_blocks_total` | Counter | Positive deltas of the rank's cumulative total-allocation snapshot. |
| `kv_cache_new_blocks_total` | Counter | Positive deltas of the rank's cumulative new-allocation snapshot. |
| `kv_cache_evictable_blocks` | Gauge | Reclaimable primary blocks summed across reported pool groups. |
| `kv_cache_peak_used_blocks` | Gauge | Sum of reported pool-group peak-used counts; an upper bound on simultaneous rank-wide usage. |
| `kv_cache_host_dropped_bytes_total` | Counter | Host-cache bytes discarded in reported iteration deltas. |
| `kv_cache_requests_suspended_total` | Counter | KV-manager suspension events in reported iteration deltas. |
| `kv_cache_requests_resumed_total` | Counter | Recovery of previously suspended requests; excludes initial admission. |
| `kv_cache_utilization` | Gauge | Used primary/GPU blocks divided by maximum blocks. |
| `kv_cache_host_utilization` | Gauge | Used secondary-pool blocks divided by its capacity, when reported. |
| `kv_cache_max_blocks` | Gauge | Maximum primary KV blocks. |
| `kv_cache_free_blocks` | Gauge | Free primary KV blocks. |
| `kv_cache_used_blocks` | Gauge | Used primary KV blocks. |
| `kv_cache_tokens_per_block` | Gauge | Tokens represented by one KV block. |
| `kv_cache_hit_rate` | Gauge | Hit rate reported in the engine's cumulative cache snapshot. |
| `kv_cache_reused_blocks_total` | Counter | Positive deltas of cumulative reused blocks, tracked independently by rank. |
| `kv_cache_missed_blocks_total` | Counter | Positive deltas of cumulative missed blocks, tracked independently by rank. |
| `kv_cache_iter_reuse_rate` | Gauge | Reused blocks divided by reused plus missed blocks in the reported iteration delta. |
| `kv_cache_iter_reused_blocks_total` | Counter | Accumulated full and partial block reuse from iteration deltas. |
| `kv_cache_iter_full_reused_blocks_total` | Counter | Fully reused blocks from iteration deltas. |
| `kv_cache_iter_partial_reused_blocks_total` | Counter | Partially reused blocks from iteration deltas. |
| `kv_cache_iter_missed_blocks_total` | Counter | Context-phase block misses from iteration deltas. |
| `kv_cache_iter_connector_reused_blocks_total` | Counter | Connector-loaded reused blocks, a subset of reused blocks; counted when a load is accepted. |
| `kv_cache_gen_alloc_blocks_total` | Counter | KV blocks allocated during generation. |
| `kv_cache_onboard_bytes_total` | Counter | Bytes transferred from host to GPU. |
| `kv_cache_offload_bytes_total` | Counter | Bytes transferred from GPU to host. |
| `kv_cache_disk_prefetch_blocks_total` | Counter | V2 blocks prefetched from disk to host. |
| `kv_cache_intra_device_copy_bytes_total` | Counter | Bytes copied within the GPU for block copies. |
| `prompt_cache_hit_tokens_total` | Counter | Prompt cache-hit tokens by `dp_rank`, `cache_level`, and `cache_tier`, accumulated from engine iteration records. |

### Memory

| Metric | Type | Meaning |
| --- | --- | --- |
| `gpu_memory_usage_bytes` | Gauge | Engine-reported GPU memory usage. |
| `cpu_memory_usage_bytes` | Gauge | Engine-reported CPU memory usage. |
| `pinned_memory_usage_bytes` | Gauge | Engine-reported pinned host memory usage. |

### Speculative decoding

| Metric | Type | Meaning |
| --- | --- | --- |
| `spec_decode_num_draft_tokens_total` | Counter | Draft tokens accumulated from rank-0 iteration records. |
| `spec_decode_num_accepted_tokens_total` | Counter | Accepted tokens accumulated from rank-0 iteration records. |
| `spec_decode_acceptance_length` | Gauge | Acceptance length from the latest rank-0 speculative statistics. |
| `spec_decode_draft_overhead` | Gauge | Draft latency divided by iteration latency, as reported by the engine. |
| `spec_decode_drafted_tokens_total` | Counter | Drafted tokens from completed-request metrics, by `token_position`. |
| `spec_decode_accepted_tokens_total` | Counter | Accepted tokens from completed-request metrics, by `token_position`. |

### Iteration and forward timings

| Metric | Type | Meaning |
| --- | --- | --- |
| `iteration_latency_seconds` | Gauge | Latest `iterLatencyMS`, converted to seconds; the batch lifecycle timing depends on scheduler mode. |
| `host_step_time_seconds` | Gauge | Latest single executor-loop wall time from `hostStepTimeMS`, matching the log's `host_step_time`. |
| `prev_device_step_time_seconds` | Gauge | Latest lagged device-step timing from `prevDeviceStepTimeMS`, matching the log's `prev_device_step_time`. |
| `gpu_forward_time_seconds` | Gauge | Latest batch-matched GPU-forward timing from `gpuForwardTimeMS`. |
| `host_step_duration_seconds` | Histogram | Exported rank-0 host-loop wall-time samples. |
| `gpu_forward_duration_seconds` | Histogram | Exported rank-0 batch-matched GPU-forward samples. |

### Configuration and OpenEngine RPCs

| Metric | Type | Meaning |
| --- | --- | --- |
| `model_config_info` | Gauge | Value 1 with model, dtype, quantization, and available model/device metadata labels. |
| `parallel_config_info` | Gauge | Value 1 with tensor, pipeline, context, and available expert parallelism plus GPU-count labels. |
| `speculative_config_info` | Gauge | Value 1 with available speculative method, draft length, and draft-model labels. |
| `kv_cache_config_info` | Gauge | Value 1 with available KV page size, reuse, memory fraction, and dtype labels. |
| `openengine_rpc_outcomes_total` | Counter | OpenEngine-only: one terminal outcome per Generate RPC, labeled by `phase` and `outcome`. |
| `openengine_metrics_errors_total` | Counter | OpenEngine-only: collection failures labeled by `source`. |

All metric names above omit `trtllm_`. Counters use `rate()` or `increase()`;
histograms expose `_bucket`, `_sum` and `_count`. Merge bucket rates before
`histogram_quantile()`; do not average worker percentiles or sum utilization ratios.

Iteration queue, memory, limits, timings and speculative decoding are rank-0
observations. Executor-loop diagnostics carry the actual producer `global_rank`.
KV and local batching carry `dp_rank`; detailed pool/tier records
currently come only from rank 0 under attention DP. Missing source data is absent,
not measured zero. Never use conversation IDs, request IDs or hashes as labels.

Transfer outcomes count once per request on each reporting rank: whole-transfer
success, terminal error, or deadline timeout. Cancellation alone is excluded.
CTX send and GEN receive each count their own endpoint. These are cumulative CPU
snapshots carried by existing rank exchange, without extra collectives. Completed-
request transfer bytes/duration describe the separate request-completion path.

Cumulative allocation/reuse snapshots and iteration deltas are separate views;
do not add them. A decreasing cumulative KV snapshot updates its baseline without
incrementing the counter. Records must arrive in producer order per rank.
Pool-group peaks need not occur simultaneously; their sum is an upper bound.
Block counts from pools with different slot sizes are not byte totals. Connector
prompt-cache hits count accepted loads; prefetched disk pages resident on host
count as host hits. Request- and iteration-sourced cache-hit totals overlap.

## Sampling and availability

`iter_perf_stats_interval > 1` samples iteration gauges and timing histograms.
Completion, new-active and KV iteration deltas carry over skipped iterations;
other counters, including speculative tokens, can undercount. Keep the interval
at 1 for complete iteration accounting. Larger
`kv_cache_config.iteration_stats_interval` delays KV updates; align it with the
performance interval. Bounded stats buffers can drop undrained records.

Scrapes read collected observations. Gauges retain the latest value and can be
stale while idle; histogram counts cover delivered samples, not all iterations.
OpenEngine drains records approximately once per second. Timing sums are not
exact engine busy time. `prev_device_step_time_seconds` lags the host loop and
cannot be subtracted from host time to estimate CPU overhead. GPU forward time
is matched to its batch. Request-completion metrics have a separate cadence and
must not be rescaled by the iteration interval.

The exporter adds CPU and exposition work. Optional KV migration CUDA timing is
disabled by default; its costs and interpretation are described in the
[migration timing section](#sampled-migration-latency).

## Tiered KV cache

KVCacheManagerV2 reports storage metrics by `dp_rank`, `cache_level`, `cache_tier`, and
`pool_group`. Pool groups are local to their level. With attention DP, detailed storage
snapshots come from rank 0. Older native builds can omit unsupported fields.

| Metric | Type | Additional labels and meaning |
| --- | --- | --- |
| `kv_cache_tier_blocks` | Gauge | `state`: disjoint free, evictable and unavailable physical slots. |
| `kv_cache_tier_bytes` | Gauge | Same states multiplied by the pool's summed slot sizes; logical capacity, excluding allocator padding. |
| `kv_cache_tier_moved_blocks_total` | Counter | `operation`: restore, evict, drop_capacity or drop_priority. |
| `kv_cache_tier_moved_bytes_total` | Counter | Movement multiplied by the attributed pool's slot size, not wire traffic. |
| `kv_cache_tier_migration_duration_seconds` | Histogram | Sampled restore/evict batches, with source/destination level and tier labels. |
| `kv_cache_migration_timing_skipped_total` | Counter | Eligible timing samples skipped at the pending-event limit. |

Free slots are unallocated. Evictable slots hold pages reclaimable by the eviction policy.
The exported `unavailable = total - free - evictable` includes retained cold-tier copies,
which can also be reclaimed and are excluded by native `StorageStatistics::unavailable()`.
Resident occupancy is `(unavailable + evictable) / total`. Keep `pool_group` in the query
to identify capacity pressure that a tier average can hide.

Restore counts accepted movement toward a hotter level on the destination pool. Evict
counts movement toward a colder level on the source pool. Capacity and priority drops are
attributed to the last resident pool. Release, cache clear, and same-level copies are excluded.
Counters measure accepted submissions. Async errors do not roll them back. Byte metrics
use the attributed pool's logical slot size, which can differ from wire bytes.

### Sampled migration latency

Set `TRTLLM_KV_CACHE_TIMING_SAMPLE_INTERVAL=N` before creating the engine. Positive N samples
the first and every Nth eligible batch per source/destination/pool path. Zero disables
timing by default. With stats enabled, the manager polls up to 64 pending CUDA event pairs.
Collection drains completed samples and counts samples skipped at capacity. A transfer
that never completes occupies a slot until teardown.

The histogram measures stream time for a migration batch, including codec and staging
work on that path. Check sample coverage and skipped counts when interpreting percentiles.

```promql
histogram_quantile(0.95,
  sum by (le, instance, source_cache_level, destination_cache_level) (
    rate(trtllm_kv_cache_tier_migration_duration_seconds_bucket{operation="restore"}[5m])
  )
)
```

### Conversations

`trtllm_kv_cache_num_conversations` counts retention-map entries, including retained drop
plans. Entries can exist without resident KV blocks. The gauge is absent when retention
tracking is disabled.

With iteration stats enabled, workers separately track admissions by
`(conversation_id, lora_task_id, cache_salt)` after request conversion and placement.
They exclude output children, dummy requests, and blank IDs. Repeated turns refresh the
last-seen time, and later cancellation leaves admission history intact. Worker counts
can overlap. Identities stay in the worker.

| Metric | Type | Meaning |
| --- | --- | --- |
| `trtllm_conversation_unique_seen{window="10m/1h"}` | Gauge | Distinct tracked identities in `(now - window, now]`; lower bound while incomplete. |
| `trtllm_conversation_unique_seen_complete{window="10m/1h"}` | Gauge | Full window observed without capacity skips. |
| `trtllm_conversation_tracking_overflow_total{window="10m/1h",reason="entries/identity_bytes"}` | Counter | Skipped admission observations, not distinct missing identities. |

The 10-minute and 1-hour windows share limits of 100,000 entries and 16 MiB of
accounted identity storage. A key in both windows consumes two entries. Container and
allocator overhead are outside the byte accounting. Known identities refresh at capacity.
Startup and skipped observations leave a window incomplete until one full window has elapsed.
Cache clears preserve activity history. Restart resets it.

Snapshots refresh at most once a second and use the existing ADP exchange. Expiring many
identities together costs up to one pass over the retained entries. Filter on
`trtllm_conversation_unique_seen_complete == 1` when queries require exact counts.
