<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prometheus performance metrics

TensorRT-LLM exposes request and engine performance metrics at
`/prometheus/metrics`. Use them to monitor latency, token throughput, queue
pressure, batching, KV-cache capacity and reuse, memory, and speculative
decoding. The HTTP server and OpenEngine gRPC server share the `trtllm_`
metric families described below.

## Enable collection

Save the following configuration as `metrics.yaml`:

```yaml
return_perf_metrics: true
enable_iter_perf_stats: true
enable_iter_req_stats: false
iter_perf_stats_interval: 1
```

Start an HTTP server and inspect its metrics:

```bash
trtllm-serve /path/to/model --config metrics.yaml --port 8000
curl --fail http://localhost:8000/prometheus/metrics
```

For OpenEngine, the metrics endpoint is a separate HTTP listener:

```bash
trtllm-serve /path/to/model --config metrics.yaml \
  --grpc --grpc-protocol openengine --port 50051
curl --fail http://localhost:8001/prometheus/metrics
```

The OpenEngine listener uses port `8001` and inherits the serving host by
default. Set `--openengine-metrics-host` and `--openengine-metrics-port` to
change its address; the metrics and gRPC ports must differ. See the
[OpenEngine serving guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/grpc/openengine/README.md#prometheus-metrics)
for its deployment and multiprocess settings.

| Setting | Effect |
| --- | --- |
| `return_perf_metrics: true` | Enables request performance collection and the Prometheus endpoint. Defaults to `false`. |
| `enable_iter_req_stats: false` | Keeps per-request iteration records disabled. Aggregated Prometheus metrics and completed-request KV-transfer metrics do not require them. |
| `enable_iter_perf_stats: true` | Adds engine iteration statistics to collection. Defaults to `false`; it does not enable the Prometheus endpoint by itself. |
| `iter_perf_stats_interval` | Builds an engine statistics record every N iterations; defaults to 1. Larger values reduce collection work and change the meaning of sampled metrics, as described below. |
| `kv_cache_config.iteration_stats_interval` | Controls how often KV iteration statistics are fetched. Larger values delay KV updates; align this interval with the iteration-performance interval. |
| `perf_metrics_output_dir` | Also enables performance collection and the endpoint. HTTP serving can write per-request JSONL; OpenEngine does not write those HTTP JSONL files. |
| `prometheus_metrics_config` | Customizes request-latency histogram buckets. Bucket values are seconds and must be nonempty and strictly increasing. |

The configurable bucket fields are `e2e_request_latency_buckets`,
`time_to_first_token_buckets`, `time_per_output_token_buckets`,
`request_queue_time_buckets`, `request_prefill_time_buckets`,
`request_decode_time_buckets`, and `request_inference_time_buckets`. For example:

```yaml
prometheus_metrics_config:
  time_to_first_token_buckets: [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
```

This example covers TTFT from 0.5 seconds through 20 seconds. Choose finite
buckets that cover your latency objectives and workload tail. Prometheus appends a `+Inf` bucket, but samples
above the largest finite boundary cannot be resolved into useful tail
percentiles; the boundary is not a request timeout.

`print_iter_log` is independent of Prometheus collection and need not be
enabled. The HTTP `/metrics` endpoint returns structured iteration statistics;
use `/prometheus/metrics` for Prometheus exposition.

## Configure scraping

Configure each serving worker as a target. For a Dynamo frontend backed by
OpenEngine, scrape the engine's HTTP metrics endpoint separately from the
Dynamo frontend. For disaggregated serving, keep prefill and decode targets
distinguishable, since their latency metrics describe different phases.

```yaml
scrape_configs:
  - job_name: trtllm-http
    scrape_interval: 15s
    metrics_path: /prometheus/metrics
    static_configs:
      - targets: ["http-worker:8000"]
  - job_name: trtllm-openengine
    scrape_interval: 15s
    metrics_path: /prometheus/metrics
    static_configs:
      - targets: ["grpc-worker:8001"]
```

Replace these example targets with reachable addresses for your deployment.
A scrape reads collected samples; it does not run inference. Restrict access
to the metrics listener to the intended monitoring network.

## Types, labels, and scope

- **Counters** accumulate events, tokens, blocks, or bytes. Use `rate()` for
  per-second rates and `increase()` for interval totals. A process restart can
  reset a counter; Prometheus rate functions account for resets.
- **Gauges** show the latest available measurement. They can decrease and can
  retain an old value while the engine is idle or a field is absent. Do not
  use `rate()` on load, utilization, or latency gauges.
- **Histograms** expose `_bucket`, `_sum`, and `_count` samples. Use
  `histogram_quantile()` on bucket rates for percentiles, or divide the sum
  rate by the count rate for a mean. Merge buckets before calculating a
  percentile; do not average percentiles across workers.

The normal serving labels are `model_name` and `engine_type`. Additional
labels depend on the metric family:

| Label | Applies to | Meaning |
| --- | --- | --- |
| `finished_reason` | Request success counter | Completion finish reason. |
| `http_code` | HTTP request error counter | HTTP status code. |
| `dp_rank` | Local KV-cache and batching metrics, tiered cache-hit tokens | Attention-DP rank; unranked records use 0. |
| `cache_level`, `cache_tier` | Tiered prompt cache-hit tokens | Configured level index and backing tier. Connector loads use an empty level and `connector` tier. |
| `token_position` | Per-position speculative counters | Draft position, starting at 0; at most 16 positions are exported. |
| `scheduler_mode` | Host-loop/device/GPU-forward timings | `overlap`, `non_overlap`, or `unknown` for missing or unrecognized modes. |
| `outcome` | KV-transfer request outcomes | `success`, `error`, or `timeout`; one outcome per request on each reporting rank. |
| `phase`, `outcome` | OpenEngine RPC outcomes | Execution phase and terminal RPC outcome. |

Configuration info gauges add configuration values as labels and have a
sample value of 1. HTTP multiprocess gauge exposition can include `pid`;
OpenEngine engine gauges use the latest live value without a `pid` label.
Prometheus also attaches target labels such as `job` and `instance`. Retain
`instance` when diagnosing one worker and avoid summing multiple frontend
copies of the same engine measurement.

Iteration queue, memory, limits, timing, and speculative-decoding metrics are
rank-0-owned measurements. They are not automatic sums across ranks or GPUs.
Local KV-cache and batching metrics use `dp_rank`; aggregate them only when
the units and ownership make that meaningful. In particular, sum used and
maximum blocks before computing aggregate utilization instead of summing
utilization ratios.

## Metric reference

All metric names in the tables below omit the common **`trtllm_` prefix**.
For example, `generation_tokens_total` is exposed as
`trtllm_generation_tokens_total`. Types refer to the metric family, before
histogram suffixes such as `_bucket` are added.

### Request latency, outcomes, and tokens

Request timings use seconds and are recorded when completion metrics become
available. They measure the worker's execution phase. In a disaggregated
deployment, a worker's timing is not the client's full end-to-end latency.
Requests rejected before execution or interrupted before completion may have
no latency observation.

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

KV-transfer timing and size come from `RequestPerfMetrics`, also used by
JSONL and Server-Timing. Transfer observations are request-scoped and recorded
only for the first output candidate, so multiple outputs do not multiply the
same transfer. These families use model labels, not a fabricated rank label:
the request's reported transfer size can already aggregate across ranks.
For disaggregated serving, retain the worker role in scrape target labels to
separate observations from the prefill and decode sides. Duration requires
valid increasing timestamps; byte totals require a positive reported size.
This path needs `return_perf_metrics`, not `enable_iter_req_stats`, and is
independent of iteration sampling.

Prompt-cache counters are process-lifetime totals, not current cache contents;
eviction or a cache flush does not clear them. Perplexity families only gain
observations when the engine supplies the corresponding measurements.

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

The request success counter and iteration completion counter have different
sources and counting scopes. Use the former for completed output candidates
and the latter for engine request completions; do not add them together.

### KV-cache capacity, reuse, and transfers

KV-cache gauges and counters below carry `dp_rank`. Cache hit and reuse rates
are ratios, while block and byte counters have different units and should
not be combined in a single rate calculation.

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

`kv_cache_transfer_requests_total` counts each request once on each reporting
rank. `success` requires the whole KV transfer (including required auxiliary
state) to complete, not a chunk, layer, task, or sender polling slice. `timeout`
means the overall transfer deadline was exceeded; a later completion or cleanup
error does not count again. Other terminal transfer failures count as `error`.
User cancellation alone is excluded. CTX send and GEN receive workers each
report their own side; summing both counts both endpoints.

Expired asynchronous GEN receives are cancelled from the transfer poll even
when no model batch can run. A refused cancellation retains the request and KV
pages and is retried; request failure follows only after safe retirement. CTX
resources are reclaimed after the final transceiver or connector owner releases
them, including requests already removed from the active list.

Counts accumulate on the executor CPU and ride the existing sampled iteration
records and attention-DP exchange as three integers, without extra collectives,
CUDA synchronization, or request iteration stats. Skipped iterations do not lose
outcomes. Export waits for a subsequent stats record, so idle workers can show
stale counts until another record is produced. Non-ADP uses the existing rank-0
export ownership. This family covers disaggregated transceiver transfers, not
external KV connector operations.

The basic allocation snapshots and context-length summaries use fields
already transported in rank-local records. Rich KV iteration details such as
reclaimable blocks, pool peaks, suspensions, and host-cache drops are exported
only where the source record supplies them. The current attention-DP fanout
provides those detailed records on rank 0; absent ranks are not emitted as
measured zeros. This exporter does not expand rank collectives to obtain them.

Pool-group peak values can occur at different times. Their sum is a pressure
indicator and upper bound, not a measured simultaneous peak. Block counts
also do not directly measure bytes when pool groups have different block sizes.

The cumulative-snapshot counters and iteration-delta counters are separate
views of reuse and misses; do not add the two views. A decrease in a rank's
cumulative reused/missed snapshot updates the collector's baseline without
adding a negative counter increment. Records must arrive in producer order
within each rank.

Tiered prompt-cache token accounting distinguishes configured levels even
when two levels use the same backing tier. Disk pages already prefetched to
host count as host hits. Connector hits are recorded when the load is
accepted, before asynchronous transfer completion. This iteration-sourced
family and the request-sourced `prompt_cached_tokens_total` can update at
different times; they are not independent token totals to add together.

### Memory

These rank-0 measurements are bytes, not GPU utilization percentages or a
sum of memory across all devices.

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

Per-position counters require those arrays in request performance metrics
and export at most positions 0 through 15. Iteration counters and
request-completion counters describe different observation streams; do not
sum them together. Use matching accepted and drafted families when
calculating an acceptance ratio.

### Iteration and forward timings

| Metric | Type | Meaning |
| --- | --- | --- |
| `iteration_latency_seconds` | Gauge | Latest `iterLatencyMS`, converted to seconds; the batch lifecycle timing depends on scheduler mode. |
| `host_step_time_seconds` | Gauge | Latest single executor-loop wall time from `hostStepTimeMS`, matching the log's `host_step_time`. |
| `prev_device_step_time_seconds` | Gauge | Latest lagged device-step timing from `prevDeviceStepTimeMS`, matching the log's `prev_device_step_time`. |
| `gpu_forward_time_seconds` | Gauge | Latest batch-matched GPU-forward timing from `gpuForwardTimeMS`. |
| `host_step_duration_seconds` | Histogram | Exported rank-0 host-loop wall-time samples. |
| `gpu_forward_duration_seconds` | Histogram | Exported rank-0 batch-matched GPU-forward samples. |

The loop/device/forward families add `scheduler_mode`. They are rank-0
measurements: attention-DP fanout copies these timing values to other rank
rows, so counting every row would duplicate observations.

With overlap scheduling, including pipeline parallelism, `iterLatencyMS`
spans the batch lifecycle, roughly two executor loops. Use host-step timing
for single-loop wall time. The lagged device-step value describes an older
loop than the host timing beside it; subtracting the gauges does not measure
CPU overhead. Batch-matched GPU-forward time also has a different scope
from host-loop wall time. Missing, null, negative, and non-finite loop/device/
forward timings do not create observations; zero is valid. Their histogram
buckets span 0.1 ms through 10 s plus the standard `+Inf` bucket.

### Configuration and OpenEngine RPCs

| Metric | Type | Meaning |
| --- | --- | --- |
| `model_config_info` | Gauge | Value 1 with model, dtype, quantization, and available model/device metadata labels. |
| `parallel_config_info` | Gauge | Value 1 with tensor, pipeline, context, and available expert parallelism plus GPU-count labels. |
| `speculative_config_info` | Gauge | Value 1 with available speculative method, draft length, and draft-model labels. |
| `kv_cache_config_info` | Gauge | Value 1 with available KV page size, reuse, memory fraction, and dtype labels. |
| `openengine_rpc_outcomes_total` | Counter | OpenEngine-only: one terminal outcome per Generate RPC, labeled by `phase` and `outcome`. |
| `openengine_metrics_errors_total` | Counter | OpenEngine-only: collection failures labeled by `source`. |

A successful RPC can complete multiple output candidates; the RPC outcome
counter and request success counter therefore need not match. An RPC's
completion means the server consumed its result stream, not that the client
acknowledged receipt. Health probes are excluded from request metrics, but
their engine work can appear in iteration statistics. A successful metrics
scrape checks the exporter; use inference health endpoints for engine health.

## Dashboard queries

Retain `instance` to compare workers. Remove it only when deliberately
aggregating across targets with compatible roles and labels.

### Output token throughput

```promql
sum by (instance, model_name) (
  rate(trtllm_generation_tokens_total[5m])
)
```

### p99 time to first token

```promql
histogram_quantile(
  0.99,
  sum by (le, instance, model_name) (
    rate(trtllm_time_to_first_token_seconds_bucket[5m])
  )
)
```

### Mean queue delay

```promql
sum by (instance, model_name) (rate(trtllm_request_queue_time_seconds_sum[5m]))
/
sum by (instance, model_name) (rate(trtllm_request_queue_time_seconds_count[5m]))
```

For current queue pressure, graph `trtllm_num_requests_waiting` alongside
`trtllm_num_requests_running` and latency; queue length alone does not show
how long requests wait.

### KV capacity across local ranks

```promql
sum by (instance, model_name) (trtllm_kv_cache_used_blocks)
/
sum by (instance, model_name) (trtllm_kv_cache_max_blocks)
```

This ratio assumes compatible block sizes across the ranks being combined.
Keep `dp_rank` to identify local capacity imbalance. For reuse over a time
window, divide the rate of `kv_cache_iter_reused_blocks_total` by the sum of
reused and missed block rates, using the same labels and counter view.

### Speculative token acceptance

```promql
sum by (instance, model_name) (rate(trtllm_spec_decode_num_accepted_tokens_total[5m]))
/
sum by (instance, model_name) (rate(trtllm_spec_decode_num_draft_tokens_total[5m]))
```

A ratio is undefined when its denominator is zero. Use the per-position
families instead when investigating which draft positions are being rejected.

### p99 host-loop time

```promql
histogram_quantile(
  0.99,
  sum by (le, instance, model_name, scheduler_mode) (
    rate(trtllm_host_step_duration_seconds_bucket[5m])
  )
)
```

## Sampling and availability

The exporter reuses existing engine/request statistics. It adds no CUDA
events, GPU synchronizations, expert histograms, request-ID labels, or rank
collectives. Its iteration updates run only for delivered sampled records;
`enable_iter_req_stats` can remain false. The existing engine instrumentation
still has a cost: `iter_perf_stats_interval` controls structured record
construction, while `print_iter_log` independently logs executor loops.
Keep iteration logging off when only Prometheus summaries are needed.

Collection and scraping have separate cadences. The servers drain available
iteration records in the background, and OpenEngine polls approximately once
per second. A histogram can retain multiple observations between scrapes;
a gauge retains only the latest available value. Metric families can have
no sample until their source data arrives, and gauges need not become zero
when a worker becomes idle.

With `iter_perf_stats_interval > 1`, iteration gauges and timing/batch
histograms describe sampled iterations. The producer carries completion,
new-active-request, and KV iteration deltas from skipped iterations into a
later record; other iteration counters, including speculative draft and
accepted tokens, can undercount. Keep the interval at 1 when complete
iteration counters are required. Larger KV-fetch intervals can delay KV
updates further. Bounded statistics buffers can also drop records if they
are not drained quickly enough.

Histogram counts do not count skipped iterations, and timing sums are not
exact engine busy time. Request-completion metrics have their own collection
path and should not be rescaled by the iteration sampling interval.

The iteration log also includes routing fetch counts, dummy/unscheduled
request counts, dummy token counts, and model state details that are not all
in the structured statistics. Exposing these requires iteration-matched,
rank-local producer data. Iteration IDs, timestamps, and free-form model
states should not be Prometheus labels.
